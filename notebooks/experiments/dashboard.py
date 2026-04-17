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

def call_writer_agent(query_metadata, similar_cases, local_fraud_rate, ml_score, question):
    """Call writer agent and return narrative string."""
    payload = {
        "query_metadata": query_metadata,
        "similar_cases": similar_cases,
        "local_fraud_rate": local_fraud_rate,
        "risk_score": ml_score if ml_score is not None else None,
        "shap_values": None,
        "question": question if question and question.strip() else None
    }
    try:
        resp = requests.post(WRITER_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json().get("narrative", "No narrative generated.")
    except Exception as e:
        return f"Error generating narrative: {e}"

def display_analysis_summary(ret_data, ml_score):
    """Display metrics, risk level, and similar cases table."""
    similar_cases = ret_data["similar_cases"]
    local_fraud_rate = ret_data["local_fraud_rate"]
    total_neighbors = ret_data["total_neighbors"]
    fraud_count = sum(1 for c in similar_cases if c["fraud_bool"] == 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Local Fraud Rate", f"{local_fraud_rate:.2%}", 
                help=f"{fraud_count} out of {total_neighbors} similar cases were fraudulent")
    col2.metric("Total Similar Cases", total_neighbors)
    col3.metric("ML Risk Score", f"{ml_score:.2f}" if ml_score is not None else "Not provided")
    
    if local_fraud_rate < 0.2:
        risk_level = "🟢 Low Risk"
        recommendation = "Approve"
    elif local_fraud_rate < 0.5:
        risk_level = "🟡 Medium Risk"
        recommendation = "Review"
    else:
        risk_level = "🔴 High Risk"
        recommendation = "Reject"
    st.info(f"**Risk Level:** {risk_level} | **Recommendation:** {recommendation}")
    
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
    use_ml = st.checkbox("Include ML risk score?")
    ml_score = None
    if use_ml:
        ml_score = st.slider("ML risk score (0-1)", 0.0, 1.0, 0.85, 0.01)
    
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
                    ml_score,
                    None   # empty question → full report
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
    st.caption("**Instructions**\n- Enter an Application ID and click 'New Analysis'.\n- Ask follow‑up questions in the chat box below.\n- Each follow‑up uses the same cached similar cases (no refetch).\n- To analyse a different ID, click 'New Analysis' again.")

# ------------------------------------------------------------------
# Main area: chat interface and analysis display
# ------------------------------------------------------------------
if st.session_state.retriever_data is not None:
    # Display analysis summary (metrics + table) always on top
    display_analysis_summary(st.session_state.retriever_data, ml_score)
    
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
                    ml_score,
                    prompt
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_narrative = answer
else:
    st.info("👈 Enter an Application ID in the sidebar and click 'New Analysis' to start.")