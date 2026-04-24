import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import google.generativeai as genai

# ============================================
# GEMINI EXPLANATION FUNCTION
# ============================================

def get_gemini_explanation(di, rate_priv, rate_unpriv, privileged_group, unprivileged_group):
    """
    Calls Gemini API to generate a plain‑English explanation of bias metrics.
    Requires API key stored in Streamlit secrets.
    """
    try:
        genai.configure(api_key=st.secrets["AIzaSyC5paZU1Ki4Hehou1nM2TDbAgDapD0APuA"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
You are a fairness auditor. A dataset shows:
- Disparate Impact (DI) = {di:.3f}
- Favorable outcome rate for {privileged_group} = {rate_priv:.2%}
- Favorable outcome rate for {unprivileged_group} = {rate_unpriv:.2%}

Interpretation guidelines:
- DI < 0.8 means bias against unprivileged group.
- DI > 1.25 means reverse bias (privileged group disadvantaged).
- 0.8 ≤ DI ≤ 1.25 is considered fair.

Provide a short (max 150 words) explanation for a non‑technical manager. Include:
1. Whether bias exists and in which direction.
2. A real‑world consequence (e.g., in hiring or loans).
3. One practical mitigation suggestion.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini explanation failed: {e}. Check API key and internet connection."

# ============================================
# MANUAL FAIRNESS METRICS (no fairlearn needed)
# ============================================

def demographic_parity_difference(y_true, y_pred, sensitive_features):
    groups = np.unique(sensitive_features)
    rates = []
    for g in groups:
        mask = (sensitive_features == g)
        if np.sum(mask) > 0:
            rate = np.mean(y_pred[mask])
            rates.append(rate)
    return np.max(rates) - np.min(rates)

def equalized_odds_difference(y_true, y_pred, sensitive_features):
    groups = np.unique(sensitive_features)
    tprs = []
    fprs = []
    for g in groups:
        mask = (sensitive_features == g)
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        tp = np.sum((y_pred_g == 1) & (y_true_g == 1))
        fn = np.sum((y_pred_g == 0) & (y_true_g == 1))
        fp = np.sum((y_pred_g == 1) & (y_true_g == 0))
        tn = np.sum((y_pred_g == 0) & (y_true_g == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    return max(np.max(tprs) - np.min(tprs), np.max(fprs) - np.min(fprs))

# ============================================
# BIAS DETECTION FUNCTIONS
# ============================================

def calculate_disparate_impact(df, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_outcome):
    if df[outcome_col].dtype == 'object':
        target = str(favorable_outcome)
    else:
        try:
            target = df[outcome_col].dtype.type(favorable_outcome)
        except:
            target = favorable_outcome

    privileged = df[df[sensitive_col] == privileged_group]
    unprivileged = df[df[sensitive_col] == unprivileged_group]

    rate_privileged = (privileged[outcome_col] == target).mean()
    rate_unprivileged = (unprivileged[outcome_col] == target).mean()

    if rate_privileged == 0:
        st.error(f"❌ No matching favorable value '{favorable_outcome}' found in column '{outcome_col}'. Unique values: {df[outcome_col].dropna().unique()}")
        st.stop()

    di = rate_unprivileged / rate_privileged
    return di, rate_privileged, rate_unprivileged

def get_bias_flag(di):
    if di < 0.8:
        return "🔴 HIGH BIAS", "Unprivileged group is disadvantaged (DI < 0.8)", "red"
    elif di > 1.25:
        return "🟡 REVERSE BIAS", "Privileged group is disadvantaged (DI > 1.25)", "orange"
    else:
        return "🟢 FAIR", "No significant adverse impact (0.8 ≤ DI ≤ 1.25)", "green"

def reweight_dataset(df, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_outcome):
    df_fixed = df.copy()
    if df[outcome_col].dtype == 'object':
        target = str(favorable_outcome)
    else:
        target = favorable_outcome

    priv_mask = (df_fixed[sensitive_col] == privileged_group)
    unpriv_mask = (df_fixed[sensitive_col] == unprivileged_group)

    rate_priv = (df_fixed.loc[priv_mask, outcome_col] == target).mean()
    rate_unpriv = (df_fixed.loc[unpriv_mask, outcome_col] == target).mean()

    if rate_unpriv >= rate_priv:
        return df_fixed, "No reweighting needed (unprivileged rate already higher or equal)"

    priv_favorable = df_fixed[(priv_mask) & (df_fixed[outcome_col] == target)]
    unpriv_favorable = df_fixed[(unpriv_mask) & (df_fixed[outcome_col] == target)]

    if len(priv_favorable) == 0 or len(unpriv_favorable) == 0:
        return df_fixed, "Cannot reweight – one group has zero favorable outcomes."

    target_unpriv_favorable_count = int(len(priv_favorable) * (rate_priv / rate_unpriv))
    needed = max(0, target_unpriv_favorable_count - len(unpriv_favorable))

    if needed > 0:
        additional = unpriv_favorable.sample(n=min(needed, len(unpriv_favorable)), replace=True)
        df_fixed = pd.concat([df_fixed, additional], ignore_index=True)

    return df_fixed, f"Added {needed} rows from unprivileged favorable group."

def threshold_adjustment(df, sensitive_col, privileged_group, unprivileged_group, score_col, favorable_outcome):
    df_fixed = df.copy()
    if score_col not in df.columns:
        return df_fixed, "Score column not found."

    if df[outcome_col].dtype == 'object':
        target = str(favorable_outcome)
    else:
        target = favorable_outcome

    unpriv = df_fixed[df_fixed[sensitive_col] == unprivileged_group]
    priv = df_fixed[df_fixed[sensitive_col] == privileged_group]

    current_rate_priv = (priv[outcome_col] == target).mean()
    unpriv_sorted = unpriv.sort_values(score_col, ascending=False)
    target_count = int(current_rate_priv * len(unpriv))

    if 0 < target_count <= len(unpriv):
        threshold = unpriv_sorted.iloc[min(target_count, len(unpriv_sorted)-1)][score_col]
        new_approvals = df_fixed[score_col] >= threshold
        df_fixed.loc[df_fixed[sensitive_col] == unprivileged_group, outcome_col] = new_approvals.loc[df_fixed[sensitive_col] == unprivileged_group].astype(int)
        message = f"Adjusted threshold to {threshold:.3f}. Unprivileged group now has same approval rate ({current_rate_priv:.2%}) as privileged."
    else:
        message = "Cannot adjust threshold – target count out of range."

    return df_fixed, message

# ============================================
# MODEL INSPECTION FUNCTIONS
# ============================================

def load_model(uploaded_file):
    return pickle.loads(uploaded_file.read())

def calculate_model_fairness_metrics(model, X_test, y_test, sensitive_attr):
    y_pred = model.predict(X_test)
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_attr)
    eq_odds_diff = equalized_odds_difference(y_test, y_pred, sensitive_attr)
    acc = accuracy_score(y_test, y_pred)
    return {"Demographic Parity Difference": dp_diff, "Equalized Odds Difference": eq_odds_diff, "Accuracy": acc}, y_pred

def postprocess_fairness(y_pred, sensitive_attr, privileged_group, unprivileged_group, flip_ratio=0.1):
    y_fixed = y_pred.copy()
    unpriv_mask = (sensitive_attr == unprivileged_group)
    n_flip = int(np.sum(unpriv_mask) * flip_ratio)
    flip_indices = np.where(unpriv_mask)[0][:n_flip]
    y_fixed[flip_indices] = 1 - y_fixed[flip_indices]
    return y_fixed

# ============================================
# STREAMLIT UI WITH SESSION STATE
# ============================================

st.set_page_config(page_title="FairLens - Bias Detector & Fixer", layout="wide")
st.title("🔍 FairLens: Bias Detection & Mitigation")
st.markdown("Upload your dataset, detect hidden discrimination, and fix it with one click.")

# Session state initialisation (same as before)
if "df" not in st.session_state:
    st.session_state.df = None
if "sensitive_col" not in st.session_state:
    st.session_state.sensitive_col = None
if "outcome_col" not in st.session_state:
    st.session_state.outcome_col = None
if "privileged_group" not in st.session_state:
    st.session_state.privileged_group = None
if "unprivileged_group" not in st.session_state:
    st.session_state.unprivileged_group = None
if "favorable_outcome" not in st.session_state:
    st.session_state.favorable_outcome = "1"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "di" not in st.session_state:
    st.session_state.di = None
if "rate_priv" not in st.session_state:
    st.session_state.rate_priv = None
if "rate_unpriv" not in st.session_state:
    st.session_state.rate_unpriv = None
if "outcome_col_enc" not in st.session_state:
    st.session_state.outcome_col_enc = None
if "favorable_encoded" not in st.session_state:
    st.session_state.favorable_encoded = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None

# Model inspection session state
if "model_uploaded" not in st.session_state:
    st.session_state.model_uploaded = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_sensitive_col" not in st.session_state:
    st.session_state.model_sensitive_col = None
if "model_outcome_col" not in st.session_state:
    st.session_state.model_outcome_col = None
if "model_privileged" not in st.session_state:
    st.session_state.model_privileged = None
if "model_unprivileged" not in st.session_state:
    st.session_state.model_unprivileged = None
if "model_fav_outcome" not in st.session_state:
    st.session_state.model_fav_outcome = "1"
if "model_evaluated" not in st.session_state:
    st.session_state.model_evaluated = False
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "y_pred_original" not in st.session_state:
    st.session_state.y_pred_original = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "sensitive_attr_values" not in st.session_state:
    st.session_state.sensitive_attr_values = None

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        if st.session_state.df is None or st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.original_df = st.session_state.df.copy()
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.analysis_done = False
            st.session_state.model_evaluated = False
        st.success(f"Loaded {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")

        if st.session_state.df is not None:
            st.session_state.sensitive_col = st.selectbox(
                "Sensitive Attribute (e.g., gender, race)",
                st.session_state.df.columns,
                index=st.session_state.df.columns.get_loc(st.session_state.sensitive_col) if st.session_state.sensitive_col in st.session_state.df.columns else 0,
                key="sensitive_select"
            )
            st.session_state.outcome_col = st.selectbox(
                "Outcome Column (e.g., approved, hired)",
                st.session_state.df.columns,
                index=st.session_state.df.columns.get_loc(st.session_state.outcome_col) if st.session_state.outcome_col in st.session_state.df.columns else 0,
                key="outcome_select"
            )
            if st.session_state.outcome_col:
                unique_vals = st.session_state.df[st.session_state.outcome_col].dropna().unique()
                st.caption(f"📌 Unique values: {', '.join(str(v) for v in unique_vals)}")

            unique_vals = st.session_state.df[st.session_state.sensitive_col].unique()
            st.session_state.privileged_group = st.selectbox(
                "Privileged Group",
                unique_vals,
                index=list(unique_vals).index(st.session_state.privileged_group) if st.session_state.privileged_group in unique_vals else 0,
                key="priv_select"
            )
            st.session_state.unprivileged_group = st.selectbox(
                "Unprivileged Group",
                [v for v in unique_vals if v != st.session_state.privileged_group],
                key="unpriv_select"
            )
            st.session_state.favorable_outcome = st.text_input(
                "Favorable outcome value (exact match, case‑sensitive)",
                value=str(st.session_state.favorable_outcome),
                key="fav_outcome"
            )

            if st.button("🚀 Run Bias Analysis", key="run_analysis"):
                df = st.session_state.df
                if df[st.session_state.outcome_col].dtype != 'object':
                    try:
                        favorable_val = int(st.session_state.favorable_outcome)
                    except ValueError:
                        try:
                            favorable_val = float(st.session_state.favorable_outcome)
                        except ValueError:
                            favorable_val = st.session_state.favorable_outcome
                else:
                    favorable_val = st.session_state.favorable_outcome

                di, rate_priv, rate_unpriv = calculate_disparate_impact(
                    df, st.session_state.sensitive_col, st.session_state.privileged_group,
                    st.session_state.unprivileged_group, st.session_state.outcome_col, favorable_val
                )
                st.session_state.di = di
                st.session_state.rate_priv = rate_priv
                st.session_state.rate_unpriv = rate_unpriv
                st.session_state.favorable_encoded = favorable_val
                st.session_state.outcome_col_enc = st.session_state.outcome_col
                st.session_state.analysis_done = True
                st.rerun()

# ============================================
# MAIN CONTENT
# ============================================

if st.session_state.analysis_done:
    di = st.session_state.di
    rate_priv = st.session_state.rate_priv
    rate_unpriv = st.session_state.rate_unpriv
    sensitive_col = st.session_state.sensitive_col
    privileged_group = st.session_state.privileged_group
    unprivileged_group = st.session_state.unprivileged_group
    outcome_col = st.session_state.outcome_col_enc
    favorable_val = st.session_state.favorable_encoded
    df = st.session_state.df

    st.subheader("📊 Bias Analysis Report")
    flag, message, color = get_bias_flag(di)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Disparate Impact", f"{di:.3f}")
    with col2:
        st.metric(f"Privileged ({privileged_group}) Rate", f"{rate_priv:.2%}")
    with col3:
        st.metric(f"Unprivileged ({unprivileged_group}) Rate", f"{rate_unpriv:.2%}")

    st.markdown(f"## {flag}")
    st.markdown(f"<p style='color:{color}; font-size:18px'>{message}</p>", unsafe_allow_html=True)

    # ----- GEMINI EXPLANATION BUTTON -----
    st.markdown("---")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🤖 Explain with Gemini (AI)", key="gemini_btn"):
            with st.spinner("Asking Gemini for an explanation..."):
                explanation = get_gemini_explanation(di, rate_priv, rate_unpriv, privileged_group, unprivileged_group)
                st.markdown("### 📝 Gemini's Explanation")
                st.info(explanation)
    # ------------------------------------

    with st.expander("📖 What does Disparate Impact mean?"):
        st.markdown("""
        **Disparate Impact (DI)** = Rate for unprivileged group / Rate for privileged group.
        - DI < 0.8 → Bias against unprivileged group.
        - 0.8 ≤ DI ≤ 1.25 → Fair.
        - DI > 1.25 → Reverse bias.
        """)

    st.subheader("📋 Outcome Rates by Group")
    rates = df.groupby(sensitive_col)[outcome_col].apply(lambda x: (x == favorable_val).mean())
    rates_df = pd.DataFrame({"Favorable Outcome Rate": rates}).reset_index()
    rates_df["Favorable Outcome Rate"] = rates_df["Favorable Outcome Rate"].apply(lambda x: f"{x:.2%}")
    st.table(rates_df)

    # --- MODEL INSPECTION (unchanged, but for brevity I keep it compact) ---
    st.markdown("---")
    st.subheader("🤖 Model Inspection (Optional)")
    st.markdown("Upload a trained ML model and test dataset to check for algorithmic bias.")
    with st.expander("📦 Upload Model & Test Data"):
        model_file = st.file_uploader("Upload trained model (pickle file)", type=["pkl", "pickle"], key="model_uploader")
        test_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"], key="test_uploader")

        if model_file is not None:
            if st.session_state.get("last_model_name") != model_file.name:
                st.session_state.model = load_model(model_file)
                st.session_state.last_model_name = model_file.name
                st.session_state.model_evaluated = False
                st.success("Model loaded.")
        if test_file is not None:
            if st.session_state.get("last_test_name") != test_file.name:
                st.session_state.test_df = pd.read_csv(test_file)
                st.session_state.last_test_name = test_file.name
                st.session_state.model_evaluated = False
                st.success(f"Test set: {len(st.session_state.test_df)} rows.")

        if st.session_state.model is not None and st.session_state.test_df is not None:
            test_df = st.session_state.test_df
            st.session_state.model_sensitive_col = st.selectbox("Sensitive column", test_df.columns, key="model_sensitive_select")
            st.session_state.model_outcome_col = st.selectbox("Outcome column", test_df.columns, key="model_outcome_select")
            unique_vals = test_df[st.session_state.model_sensitive_col].unique()
            st.session_state.model_privileged = st.selectbox("Privileged group", unique_vals, key="model_priv_select")
            st.session_state.model_unprivileged = st.selectbox("Unprivileged group", [v for v in unique_vals if v != st.session_state.model_privileged], key="model_unpriv_select")
            unique_outcome_vals = test_df[st.session_state.model_outcome_col].dropna().unique()
            st.caption(f"Unique outcome values: {', '.join(str(v) for v in unique_outcome_vals)}")
            st.session_state.model_fav_outcome = st.text_input("Favorable outcome value", value=st.session_state.model_fav_outcome, key="model_fav_input")

            if st.button("🔍 Evaluate Model Fairness", key="eval_model_btn"):
                with st.spinner("Evaluating..."):
                    try:
                        feature_cols = [c for c in test_df.columns if c not in [st.session_state.model_sensitive_col, st.session_state.model_outcome_col]]
                        X_test = test_df[feature_cols]
                        if test_df[st.session_state.model_outcome_col].dtype == 'object':
                            fav_target = str(st.session_state.model_fav_outcome)
                        else:
                            try:
                                fav_target = int(st.session_state.model_fav_outcome)
                            except ValueError:
                                fav_target = float(st.session_state.model_fav_outcome) if '.' in st.session_state.model_fav_outcome else st.session_state.model_fav_outcome
                        y_test = (test_df[st.session_state.model_outcome_col] == fav_target).astype(int)
                        sensitive_attr = test_df[st.session_state.model_sensitive_col]
                        metrics, y_pred = calculate_model_fairness_metrics(st.session_state.model, X_test, y_test, sensitive_attr)
                        st.session_state.model_metrics = metrics
                        st.session_state.y_pred_original = y_pred
                        st.session_state.y_test = y_test
                        st.session_state.sensitive_attr_values = sensitive_attr
                        st.session_state.model_evaluated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.model_evaluated:
        m = st.session_state.model_metrics
        st.subheader("Model Fairness Report")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['Accuracy']:.2%}")
        c2.metric("Demographic Parity Diff", f"{m['Demographic Parity Difference']:.3f}")
        c3.metric("Equalized Odds Diff", f"{m['Equalized Odds Difference']:.3f}")
        if m['Demographic Parity Difference'] > 0.1:
            st.warning("⚠️ Demographic parity difference > 0.1")
        else:
            st.success("✅ Acceptable demographic parity")

        st.subheader("Post-Processing Fix")
        flip_ratio = st.slider("Flip ratio for unprivileged group", 0.0, 0.5, 0.1, 0.05, key="flip_slider")
        if st.button("Apply Post-Processing", key="postprocess_btn"):
            with st.spinner("Applying..."):
                y_fixed = postprocess_fairness(st.session_state.y_pred_original, st.session_state.sensitive_attr_values,
                                               st.session_state.model_privileged, st.session_state.model_unprivileged, flip_ratio)
                dp_fixed = demographic_parity_difference(st.session_state.y_test, y_fixed, st.session_state.sensitive_attr_values)
                eq_fixed = equalized_odds_difference(st.session_state.y_test, y_fixed, st.session_state.sensitive_attr_values)
                acc_fixed = accuracy_score(st.session_state.y_test, y_fixed)
                st.write("**After fix:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("New Accuracy", f"{acc_fixed:.2%}", delta=f"{acc_fixed - m['Accuracy']:.2%}")
                col2.metric("New Demo. Parity", f"{dp_fixed:.3f}", delta=f"{dp_fixed - m['Demographic Parity Difference']:.3f}")
                col3.metric("New Equal. Odds", f"{eq_fixed:.3f}", delta=f"{eq_fixed - m['Equalized Odds Difference']:.3f}")
                if st.session_state.test_df is not None:
                    pred_df = st.session_state.test_df.copy()
                    pred_df['original_pred'] = st.session_state.y_pred_original
                    pred_df['fixed_pred'] = y_fixed
                    csv_fixed = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download fixed predictions", csv_fixed, "fixed_preds.csv", "text/csv")

    # --- DATA FIX SECTION ---
    if di < 0.8 or di > 1.25:
        st.markdown("---")
        st.subheader("🔧 Fix the Detected Bias in Data")
        fix_method = st.radio("Mitigation method", ["Reweighting (adjust dataset)", "Threshold Adjustment (decision rule)"], key="fix_method_radio")
        if fix_method == "Reweighting (adjust dataset)":
            if st.button("Apply Reweighting", key="reweight_btn"):
                with st.spinner("Reweighting..."):
                    df_fixed, msg = reweight_dataset(df, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_val)
                    st.success(msg)
                    st.session_state.df = df_fixed
                    di_fixed, rp, ru = calculate_disparate_impact(df_fixed, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_val)
                    st.session_state.di = di_fixed
                    st.session_state.rate_priv = rp
                    st.session_state.rate_unpriv = ru
                    st.rerun()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                score_col = st.selectbox("Score column", numeric_cols, key="score_col_select")
                if st.button("Apply Threshold Adjustment", key="threshold_btn"):
                    with st.spinner("Adjusting..."):
                        df_fixed, msg = threshold_adjustment(df, sensitive_col, privileged_group, unprivileged_group, score_col, favorable_val)
                        st.info(msg)
                        st.session_state.df = df_fixed
                        di_fixed, rp, ru = calculate_disparate_impact(df_fixed, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_val)
                        st.session_state.di = di_fixed
                        st.session_state.rate_priv = rp
                        st.session_state.rate_unpriv = ru
                        st.rerun()
            else:
                st.error("No numeric columns for threshold adjustment. Use reweighting.")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Current Dataset (Fixed)", csv, "fairlens_fixed.csv", "text/csv", key="download_fixed")
        if st.button("Reset to Original Data", key="reset_btn"):
            st.session_state.df = st.session_state.original_df.copy()
            di_orig, rp, ru = calculate_disparate_impact(st.session_state.df, sensitive_col, privileged_group, unprivileged_group, outcome_col, favorable_val)
            st.session_state.di = di_orig
            st.session_state.rate_priv = rp
            st.session_state.rate_unpriv = ru
            st.rerun()
    else:
        st.success("🎉 Dataset is already fair! No fix needed.")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Dataset", csv, "fair_dataset.csv", "text/csv")

elif st.session_state.df is not None and not st.session_state.analysis_done:
    st.info("👈 Click 'Run Bias Analysis' to start.")
else:
    st.info("👈 Upload a CSV file to begin")
    with st.expander("📁 Generate a biased sample dataset"):
        st.code("""
import pandas as pd
import numpy as np
np.random.seed(42)
n = 1000
gender = np.random.choice(['Male','Female'], n)
credit_score = np.random.normal(650, 50, n)
prob_approve = 1/(1+np.exp(-(credit_score-600)/100))
prob_approve[gender=='Female'] *= 0.7
approved = np.random.binomial(1, prob_approve)
df = pd.DataFrame({'gender':gender,'credit_score':credit_score,'approved':approved})
df.to_csv('loans_biased.csv', index=False)
print("Created loans_biased.csv")
""")
