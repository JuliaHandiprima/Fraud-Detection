import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# Configuration
RETRIEVER_URL = "http://localhost:5001/retrieve"
WRITER_URL = "http://localhost:5004/write"

st.set_page_config(page_title="Fraud Detection Assistant", layout="wide")
st.title("🕵️ Fraud Detection Assistant")

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
if "retriever_data" not in st.session_state:
    st.session_state.retriever_data = None          # cached retriever output
if "current_app_id" not in st.session_state:
    st.session_state.current_app_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []              # list of {"role": "user"/"assistant", "content": ...}
if "last_narrative" not in st.session_state:
    st.session_state.last_narrative = ""

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def fetch_retriever_data(app_id):
    """Call retriever agent and return parsed JSON or None."""
    try:
        resp = requests.post(RETRIEVER_URL, json={"application_id": int(app_id)}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            st.error(f"Retriever error: {data['error']}")
            return None
        return data
    except Exception as e:
        st.error(f"Error calling Retriever Agent: {e}")
        return None

def _p_final_for_writer(ml_explanation):
    """Single scalar for writer `risk_score` field: champion p_final when available."""
    if not ml_explanation:
        return None
    rs = ml_explanation.get("risk_scores") or {}
    p = rs.get("p_final")
    return float(p) if p is not None else None


def call_writer_agent(query_metadata, similar_cases, local_fraud_rate, question, ml_explanation=None):
    """Call writer agent. Retriever supplies local_fraud_rate; champion SHAP + p_final via ml_explanation."""
    payload = {
        "query_metadata": query_metadata,
        "similar_cases": similar_cases,
        "local_fraud_rate": local_fraud_rate,
        "risk_score": _p_final_for_writer(ml_explanation),
        "ml_explanation": ml_explanation,
        "question": question if question and question.strip() else None,
    }
    try:
        resp = requests.post(WRITER_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json().get("narrative", "No narrative generated.")
    except Exception as e:
        return f"Error generating narrative: {e}"

def display_analysis_summary(ret_data):
    """Retriever local fraud rate + champion p_final (separate); optional SHAP."""
    similar_cases = ret_data["similar_cases"]
    local_fraud_rate = ret_data["local_fraud_rate"]
    total_neighbors = ret_data["total_neighbors"]
    fraud_count = sum(1 for c in similar_cases if c["fraud_bool"] == 1)
    ml_explanation = ret_data.get("ml_explanation") or {}
    rs = (ml_explanation.get("risk_scores") or {}) if isinstance(ml_explanation, dict) else {}
    p_final = rs.get("p_final")
    conf_band = rs.get("confidence_band")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Local fraud rate (retriever)",
        f"{local_fraud_rate:.2%}",
        help=f"Share of fraud labels among the {total_neighbors} nearest past applications (pgvector neighbors only). Not mixed with ML.",
    )
    col2.metric("Total similar cases", total_neighbors)
    if p_final is not None:
        col3.metric(
            "Champion p_final (ML)",
            f"{float(p_final):.4f}",
            help="XGBoost + Platt calibration + risk calibrator on retriever-enriched features. Separate from local fraud rate.",
        )
    else:
        col3.metric(
            "Champion p_final (ML)",
            "—",
            help="Set CHAMPION_MANIFEST_PATH when starting the retriever (repo root).",
        )

    if ml_explanation.get("shap_available") and ml_explanation.get("top_shap_drivers"):
        st.subheader("📊 SHAP — top drivers (this application)")
        st.caption("Positive SHAP pushes the champion model toward fraud; values from TreeExplainer on the enriched feature row.")
        shap_df = pd.DataFrame(ml_explanation["top_shap_drivers"])
        st.dataframe(shap_df, use_container_width=True)
    elif ml_explanation.get("shap_error") or ml_explanation.get("shap_note"):
        st.warning(f"SHAP / champion scores unavailable: {ml_explanation.get('shap_error') or ml_explanation.get('shap_note')}")
    
    # Retriever-only band (neighborhood)
    if local_fraud_rate < 0.2:
        retr_level = "🟢 Low"
        retr_hint = "few frauds among neighbors"
    elif local_fraud_rate < 0.5:
        retr_level = "🟡 Medium"
        retr_hint = "mixed neighbors"
    else:
        retr_level = "🔴 High"
        retr_hint = "many frauds among neighbors"

    retr_line = f"**Retriever:** {retr_level} neighborhood fraud rate ({retr_hint})."

    if p_final is not None:
        band = conf_band or "unknown"
        pol = rs.get("recommended_action") or "—"
        model_line = f"**Champion model:** p_final = **{float(p_final):.4f}** (confidence: {band}; policy hint: {pol})."
    else:
        model_line = "**Champion model:** not loaded — start retriever with `CHAMPION_MANIFEST_PATH` for p_final."

    st.info(f"{retr_line}\n\n{model_line}")

    # Build table of similar cases (top 20)
    all_features = ['income', 'velocity_6h', 'payment_type', 'prev_address_months_count', 
                    'intended_balcon_amount', 'name_email_similarity', 'zip_count_4w']
    selected_features = st.multiselect("Show features in table", all_features, default=all_features[:3], key="table_features")
    if similar_cases:
        df_data = []
        for case in similar_cases:
            meta = case.get("metadata", {})
            row = {
                "ID": case["id"],
                "Fraud": "⚠️ Yes" if case["fraud_bool"] else "✅ No",
                "Month": case["month"],
                "Similarity": f"{case['similarity']:.3f}"
            }
            for feat in selected_features:
                row[feat] = meta.get(feat, "?")
            df_data.append(row)
        df = pd.DataFrame(df_data)
        st.subheader("📋 Most Similar Past Applications")
        st.dataframe(df, use_container_width=True, height=400)
        
        with st.expander("🔍 View Full Metadata of a Similar Case"):
            case_id = st.selectbox("Select Case ID", [c["id"] for c in similar_cases], key="meta_select")
            selected_meta = next(c["metadata"] for c in similar_cases if c["id"] == case_id)
            st.json(selected_meta)
    else:
        st.warning("No similar cases found.")

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Analysis Controls")
    app_id = st.number_input("Application ID", min_value=1, step=1, value=856982, key="app_id_input")

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("🔄 New Analysis", use_container_width=True):
        # Fetch new data and reset chat history for the new app
        with st.spinner("Fetching similar cases..."):
            new_data = fetch_retriever_data(app_id)
        if new_data:
            st.session_state.retriever_data = new_data
            st.session_state.current_app_id = app_id
            st.session_state.chat_history = []   # clear chat history for new app
            st.session_state.last_narrative = ""
            # Automatically generate default report (empty question) for the new app
            with st.spinner("Generating initial report..."):
                narrative = call_writer_agent(
                    new_data["query_metadata"],
                    new_data["similar_cases"],
                    new_data["local_fraud_rate"],
                    None,
                    ml_explanation=new_data.get("ml_explanation"),
                )
                st.session_state.last_narrative = narrative
                st.session_state.chat_history.append({"role": "assistant", "content": narrative})
            st.rerun()
    
    if col_btn2.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.retriever_data = None
        st.session_state.current_app_id = None
        st.session_state.chat_history = []
        st.session_state.last_narrative = ""
        st.rerun()
    
    st.divider()
    st.caption(
        "**Instructions**\n"
        "- Enter an Application ID and click 'New Analysis'.\n"
        "- Run the retriever from the **repo root** with `CHAMPION_MANIFEST_PATH` set (e.g. `results/variants/champion_model.json`) "
        "and optional `CHAMPION_VARIANT_NAME=variant_base` so **SHAP** and **p_final** appear here and in the writer.\n"
        "- Ask follow‑up questions below; each uses the same cached retriever payload.\n"
        "- To analyse a different ID, click 'New Analysis' again."
    )

# ------------------------------------------------------------------
# Main area: chat interface and analysis display
# ------------------------------------------------------------------
if st.session_state.retriever_data is not None:
    # Display analysis summary (metrics + table) always on top
    display_analysis_summary(st.session_state.retriever_data)
    
    st.divider()
    st.subheader("💬 Ask Follow‑up Questions")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input for follow‑up questions
    if prompt := st.chat_input("Ask a follow‑up question..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer using cached data
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ret_data = st.session_state.retriever_data
                answer = call_writer_agent(
                    ret_data["query_metadata"],
                    ret_data["similar_cases"],
                    ret_data["local_fraud_rate"],
                    prompt,
                    ml_explanation=ret_data.get("ml_explanation"),
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_narrative = answer
else:
    st.info("👈 Enter an Application ID in the sidebar and click 'New Analysis' to start.")