import time
import streamlit as st
import pandas as pd
import joblib
import shap
from google import genai
from pathlib import Path

BASE = Path(__file__).parent.parent
df = pd.read_csv(BASE / "data" / "telco_clean.csv")
model = joblib.load(BASE / "models" / "xgb_tuned.pkl")
feature_cols = joblib.load(BASE / "models" / "feature_columns_tuned.pkl")

THRESHOLD = 0.3
GEMINI_MODEL = "gemini-3.8-flash"
client = genai.Client()


def ask_gemini(prompt, tries=3):
    """Gemini busy (503) ho to 3 baar retry karta hai."""
    for i in range(tries):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)


if df["Churn"].dtype == object:
    df["churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})
else:
    df["churn_flag"] = df["Churn"]

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("Customer Churn Dashboard")

tab1, tab2, tab3 = st.tabs(["Overview", "Predict churn", "Ask your data"])

# ---------- Tab 1: Overview ----------
with tab1:
    contracts = sorted(df["Contract"].unique())
    internets = sorted(df["InternetService"].unique())
    sel_contract = st.sidebar.multiselect("Contract", contracts, default=contracts)
    sel_internet = st.sidebar.multiselect("Internet Service", internets, default=internets)

    filtered = df[df["Contract"].isin(sel_contract) & df["InternetService"].isin(sel_internet)]

    if filtered.empty:
        st.warning("No data for selected filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{len(filtered):,}")
        c2.metric("Churn rate", f"{filtered['churn_flag'].mean() * 100:.1f}%")
        lost = filtered.loc[filtered["churn_flag"] == 1, "MonthlyCharges"].sum()
        c3.metric("Monthly revenue lost", f"${lost:,.0f}")

        st.subheader("Churn rate by contract type (%)")
        st.bar_chart(filtered.groupby("Contract")["churn_flag"].mean() * 100)

        st.subheader("Business impact")
        pct = st.slider("% of churning customers saved", 0, 100, 20)
        saved = lost * pct / 100
        b1, b2 = st.columns(2)
        b1.metric("Monthly revenue saved", f"${saved:,.0f}")
        b2.metric("Yearly revenue saved", f"${saved * 12:,.0f}")

        st.subheader("At-risk customers")

        score_cols = [c for c in df.columns if c not in {"Churn", "churn_flag", "customerID"}]

        @st.cache_data
        def score_all(data):
            X = pd.get_dummies(data[score_cols]).reindex(columns=feature_cols, fill_value=0).astype(float)
            return model.predict_proba(X)[:, 1]

        scored = filtered.copy()
        scored["churn_prob"] = score_all(filtered)
        at_risk = scored[scored["churn_prob"] > THRESHOLD].sort_values("churn_prob", ascending=False)

        st.write(f"{len(at_risk):,} customers above the {THRESHOLD} risk threshold")
        st.dataframe(at_risk.head(20))
        st.download_button(
            "Export At-Risk Customers to CSV",
            at_risk.to_csv(index=False).encode("utf-8"),
            file_name="at_risk_customers.csv",
            mime="text/csv",
        )

# ---------- Tab 2: Predict churn + SHAP + AI suggestion ----------
with tab2:
    st.subheader("Predict churn for a customer")
    skip = {"Churn", "churn_flag", "customerID"}
    input_cols = [c for c in df.columns if c not in skip]

    # --- Preset demo profiles ---
    presets = {
        "High-Risk": {"Contract": "Month-to-month", "tenure": 2.0, "MonthlyCharges": 95.0,
                      "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
        "Low-Risk": {"Contract": "Two year", "tenure": 60.0, "MonthlyCharges": 30.0,
                     "InternetService": "DSL", "PaymentMethod": "Bank transfer (automatic)"},
    }

    def load_preset(name):
        for col, val in presets[name].items():
            if col not in input_cols:
                continue
            if isinstance(val, str) and val not in df[col].values:
                continue
            st.session_state[f"in_{col}"] = val

    p1, p2 = st.columns(2)
    p1.button("Load High-Risk Profile", on_click=load_preset, args=("High-Risk",))
    p2.button("Load Low-Risk Profile", on_click=load_preset, args=("Low-Risk",))

    with st.form("predict_form"):
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(input_cols):
            key = f"in_{col}"
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if key not in st.session_state:
                        st.session_state[key] = float(df[col].median())
                    values[col] = st.number_input(col, key=key)
                else:
                    values[col] = st.selectbox(col, sorted(df[col].dropna().unique()), key=key)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([values])
        row = pd.get_dummies(row).reindex(columns=feature_cols, fill_value=0).astype(float)
        prob = model.predict_proba(row)[0][1]
        st.metric("Churn probability", f"{prob * 100:.1f}%")

        if prob > THRESHOLD:
            st.error("High risk of churn")
            risk = "high"
        else:
            st.success("Low risk of churn")
            risk = "low"

        # --- SHAP: model ne ye prediction kyun di ---
        st.subheader("Why this prediction? (SHAP)")
        sv = get_explainer().shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]
        contrib = pd.Series(sv[0], index=feature_cols)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)
        st.bar_chart(top)
        st.caption("Positive bars push the customer towards churn, negative bars push towards staying.")
        up_factors = ", ".join(contrib.sort_values(ascending=False).head(3).index)

        # --- AI retention suggestion ---
        st.subheader("AI retention suggestion")
        prompt = f"""You are a customer retention analyst at a telecom company.
Customer details: {values}
Predicted churn probability: {prob * 100:.1f}% ({risk} risk).
Main factors increasing churn risk (from the model): {up_factors}.
Give a retention suggestion in 2-3 short sentences: what specific offer or action
should the company take for this customer, and why. Plain text only."""
        try:
            with st.spinner("Generating suggestion..."):
                suggestion = ask_gemini(prompt)
            st.info(suggestion.replace("$", "\\$"))
        except Exception as e:
            st.warning(f"Suggestion not available: {e}")

# ---------- Tab 3: Ask your data ----------
with tab3:
    st.subheader("Ask your data")
    q = st.text_input("Ask a question about the data")
    if q:
        prompt = f"""DataFrame `df` columns: {list(df.columns)}
Sample:
{df.head(3).to_string()}
Write ONE pandas expression using df that answers: {q}
Return only the code, no explanation, no backticks."""
        try:
            code = ask_gemini(prompt).strip("`")
            st.code(code, language="python")
            st.write(eval(code, {"df": df, "pd": pd}))
        except Exception as e:
            st.error(e)