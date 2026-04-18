"""
writer_agent.py
Generates a concise, actionable fraud investigation narrative using GitHub Models.
Expects: query_metadata (dict), similar_cases (list), local_fraud_rate (float), risk_score (float or None),
optional ml_explanation (dict) with top_shap_drivers and risk_scores from champion inference.
"""

import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
# Replace the GitHub client with DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
PRIMARY_MODEL = "deepseek-chat"          # standard model (fast, good for most tasks)
FALLBACK_MODEL = "deepseek-reasoner"     # slower but stronger reasoning (optional)
# ------------------------------------------------------------------
# Helper: Compute statistical summary for a numerical feature
# ------------------------------------------------------------------
def compute_feature_stats(similar_cases, feature_name):
    fraud_vals = []
    legit_vals = []
    for case in similar_cases:
        val = case.get('metadata', {}).get(feature_name)
        if val is None or not isinstance(val, (int, float)):
            continue
        
        if case['fraud_bool'] == 1:
            fraud_vals.append(val)
        else:
            legit_vals.append(val)
    stats = {}
    if fraud_vals:
        stats['fraud_min'] = min(fraud_vals)
        stats['fraud_max'] = max(fraud_vals)
        stats['fraud_mean'] = sum(fraud_vals) / len(fraud_vals)
        stats['fraud_count'] = len(fraud_vals)
    if legit_vals:
        stats['legit_min'] = min(legit_vals)
        stats['legit_max'] = max(legit_vals)
        stats['legit_mean'] = sum(legit_vals) / len(legit_vals)
        stats['legit_count'] = len(legit_vals)
    # Check for separation
    if fraud_vals and legit_vals:
        if stats['fraud_min'] > stats['legit_max']:
            stats['separation'] = True
            stats['threshold'] = (stats['legit_max'] + stats['fraud_min']) / 2
        elif stats['fraud_max'] < stats['legit_min']:
            stats['separation'] = True
            stats['threshold'] = (stats['fraud_max'] + stats['legit_min']) / 2
        else:
            stats['separation'] = False
    else:
        stats['separation'] = False
    return stats

# ------------------------------------------------------------------
# Helper: Get most similar fraudulent and legitimate cases
# ------------------------------------------------------------------
def get_best_cases(similar_cases):
    best_fraud = None
    best_legit = None
    for case in similar_cases:
        if case['fraud_bool'] == 1:
            if best_fraud is None or case['similarity'] > best_fraud['similarity']:
                best_fraud = case
        else:
            if best_legit is None or case['similarity'] > best_legit['similarity']:
                best_legit = case
    return best_fraud, best_legit

# ------------------------------------------------------------------
# Helper: Get most common categorical values among fraudulent cases
# ------------------------------------------------------------------
def get_categorical_fraud_stats(similar_cases, feature_name):
    fraud_counts = {}
    legit_counts = {}
    for case in similar_cases:
        val = case.get('metadata', {}).get(feature_name)
        if val is None:
            continue
        if case['fraud_bool'] == 1:
            fraud_counts[val] = fraud_counts.get(val, 0) + 1
        else:
            legit_counts[val] = legit_counts.get(val, 0) + 1
    top_fraud = sorted(fraud_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    return top_fraud, fraud_counts, legit_counts

# ------------------------------------------------------------------
# Build the prompt (concise version)
# ------------------------------------------------------------------
def _format_ml_explanation(ml_explanation):
    """Turn ml_explanation dict into prompt text (SHAP + calibrated risk scores)."""
    if not ml_explanation:
        return ""
    if not ml_explanation.get("shap_available"):
        err = ml_explanation.get("shap_error") or ml_explanation.get("shap_note")
        if err:
            return f"\n**Model explanation (SHAP / risk):** unavailable ({err})\n"
        return ""

    lines = []
    rs = ml_explanation.get("risk_scores") or {}
    if rs:
        lines.append("**Calibrated risk scores (champion model):**")
        if rs.get("p_final") is not None:
            lines.append(f"- Final risk score p_final: {float(rs['p_final']):.4f} (primary score for policy)")
        if rs.get("p_model_calibrated") is not None:
            lines.append(f"- Platt-calibrated model probability: {float(rs['p_model_calibrated']):.4f}")
        if rs.get("confidence_band"):
            lines.append(f"- Confidence band: {rs['confidence_band']}")
        if rs.get("recommended_action"):
            lines.append(f"- Recommended action (policy): {rs['recommended_action']}")

    drivers = ml_explanation.get("top_shap_drivers") or []
    if drivers:
        lines.append("**SHAP top drivers (positive SHAP increases modeled fraud risk for this application):**")
        for d in drivers:
            feat = d.get("feature", "?")
            sv = d.get("shap_value", 0)
            lines.append(f"  - {feat}: {float(sv):+.4f}")

    return "\n" + "\n".join(lines) + "\n" if lines else ""


def build_prompt(query_metadata, similar_cases, local_fraud_rate, risk_score, question, ml_explanation=None):
    # ---- Numerical features (for statistical analysis) ----
    numerical_features = [
        'income', 'velocity_6h', 'intended_balcon_amount', 'prev_address_months_count',
        'current_address_months_count', 'customer_age', 'name_email_similarity',
        'zip_count_4w', 'credit_risk_score'
    ]
    # ---- Categorical features ----
    categorical_features = ['payment_type']

    # ---- Query features ----
    query_summary = {}
    for feat in numerical_features + categorical_features:
        query_summary[feat] = query_metadata.get(feat, '?')

    # ---- Statistical analysis ----
    stats_text = ""
    for feat in numerical_features:
        stats = compute_feature_stats(similar_cases, feat)
        if not stats:
            continue
        if stats.get('separation'):
            stats_text += f"- **{feat}**: Clear separation. Fraudulent cases have {feat} ≥ {stats['fraud_min']:.1f}, legitimate cases have {feat} ≤ {stats['legit_max']:.1f}. (Based on {stats['fraud_count']} fraud, {stats['legit_count']} legit)\n"
        else:
            if 'fraud_mean' in stats and 'legit_mean' in stats:
                ratio = stats['fraud_mean'] / (stats['legit_mean'] + 0.001)
                if ratio > 2 or ratio < 0.5:
                    stats_text += f"- **{feat}**: Notable difference – fraudulent average {stats['fraud_mean']:.1f}, legitimate average {stats['legit_mean']:.1f}.\n"
    if not stats_text:
        stats_text = "No strong numerical signals detected."

    # ---- Categorical patterns ----
    cat_text = ""
    for feat in categorical_features:
        top_fraud, _, _ = get_categorical_fraud_stats(similar_cases, feat)
        if top_fraud:
            cat_str = ', '.join([f"'{val}' ({cnt} cases)" for val, cnt in top_fraud])
            cat_text += f"- **{feat}**: Most common among fraudulent cases: {cat_str}\n"
    if cat_text:
        cat_text = "**Categorical patterns among fraudulent cases:**\n" + cat_text

    # ---- Best cases for comparison ----
    best_fraud, best_legit = get_best_cases(similar_cases)
    best_fraud_text = ""
    best_legit_text = ""
    table_features = ['income', 'velocity_6h', 'payment_type', 'intended_balcon_amount',
                      'zip_count_4w', 'credit_risk_score']
    if best_fraud:
        meta = best_fraud['metadata']
        best_fraud_text = f"ID={best_fraud['id']}, similarity={best_fraud['similarity']:.3f}\n"
        for feat in table_features:
            best_fraud_text += f"  {feat}: {meta.get(feat, '?')}\n"
    if best_legit:
        meta = best_legit['metadata']
        best_legit_text = f"ID={best_legit['id']}, similarity={best_legit['similarity']:.3f}\n"
        for feat in table_features:
            best_legit_text += f"  {feat}: {meta.get(feat, '?')}\n"

    # ---- Risk: retriever vs champion (champion = XGBoost + Platt + risk calibrator) ----
    rs = (ml_explanation or {}).get("risk_scores") or {}
    if rs.get("p_final") is not None:
        extra = []
        if rs.get("p_model_calibrated") is not None:
            extra.append(f"Platt-calibrated p: {float(rs['p_model_calibrated']):.4f}")
        if rs.get("confidence_band"):
            extra.append(f"band: {rs['confidence_band']}")
        if rs.get("recommended_action"):
            extra.append(f"policy: {rs['recommended_action']}")
        tail = " | ".join(extra) if extra else ""
        risk_line = (
            f"- **Champion model (XGBoost + calibration + risk calibrator):** p_final = {float(rs['p_final']):.4f}"
            + (f" ({tail})" if tail else "")
        )
    elif risk_score is not None:
        risk_line = f"- **Champion model:** not loaded; legacy risk_score field = {risk_score:.4f}"
    else:
        risk_line = "- **Champion model:** not provided (configure CHAMPION_MANIFEST_PATH on the retriever for p_final)."

    shap_block = _format_ml_explanation(ml_explanation)

    # ---- Task based on question ----
    if question and question.strip():
        task = f"Answer the following question concisely, using ONLY the data above. Do not add any extra sections (e.g., 'Top Risk Indicators', 'Comparison Table') unless the question explicitly asks for them. Question: {question}"
        extra_instructions = "- Answer ONLY the question. Do not output any default sections."
    else:
        task = """Provide a comprehensive fraud risk assessment. Follow this exact structure:

1. **Verdict & Confidence** – Clear verdict and confidence level.
2. **Query Application Features** – List key features.
3. **Comparison Table** – Compare query vs. most similar fraudulent and legitimate cases (use the provided cases).
4. **Key Risk Indicators** – List 3-5 discriminative features with explanations.
5. **Recommendation** – Clear action and next steps.
6. **Model interpretability** – If SHAP drivers and calibrated scores are provided below, summarize the top 2–3 drivers in plain language and align with the final risk score.
7. **Limitations** – Note any missing information."""
        extra_instructions = "- Follow the exact structure above. If SHAP / calibrated risk data is present, you MUST incorporate it in section 6 and cross-check with similar-case analysis."

    # ---- Assemble final prompt ----
    prompt = f"""
You are a professional bank fraud investigator.

**Query Application (key features):**
{query_summary}

{risk_line}
{shap_block}
**Retriever (neighborhood fraud rate among similar past cases):** {local_fraud_rate:.2f} — share of nearest neighbors labeled fraud; independent of the champion score above.
**Number of similar cases:** {len(similar_cases)}

**Statistical signals:**
{stats_text}

{cat_text}

**Most similar fraudulent case:**
{best_fraud_text if best_fraud_text else "None"}

**Most similar legitimate case:**
{best_legit_text if best_legit_text else "None"}

**Task:**
{task}

**Instructions:**
- Base your answer ONLY on the provided data.
- Be factual and concise.
- {extra_instructions}
- Output only the answer (no extra text).
"""
    return prompt

# ------------------------------------------------------------------
# Flask endpoint
# ------------------------------------------------------------------
@app.route('/write', methods=['POST'])
def write():
    data = request.json
    query_metadata = data.get('query_metadata', {})
    similar_cases = data.get('similar_cases', [])
    local_fraud_rate = data.get('local_fraud_rate', 0)
    risk_score = data.get('risk_score', None)
    question = data.get('question', None)
    ml_explanation = data.get('ml_explanation')
    if ml_explanation is None and data.get('top_shap_drivers') is not None:
        ml_explanation = {
            "shap_available": True,
            "top_shap_drivers": data.get('top_shap_drivers'),
            "risk_scores": data.get('risk_scores'),
        }

    if not similar_cases:
        return jsonify({"narrative": "No similar cases found. Cannot generate assessment."})

    prompt = build_prompt(query_metadata, similar_cases, local_fraud_rate, risk_score, question, ml_explanation=ml_explanation)

    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    narrative = None
    last_error = None

    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional bank fraud investigator. Always be concise and factual."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800  # Reduced for concise output
            )
            if hasattr(response, 'choices') and response.choices:
                narrative = response.choices[0].message.content
                break
            else:
                last_error = "API response missing 'choices'"
                continue
        except Exception as e:
            last_error = str(e)
            continue

    if narrative is None:
        narrative = f"Error generating narrative: {last_error}"

    return jsonify({"narrative": narrative})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)