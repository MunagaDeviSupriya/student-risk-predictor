import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Student Risk Dashboard", layout="wide")
st.title("🎓 AI-Based Student Academic Risk Prediction System")
st.markdown("""
This dashboard predicts student academic risk using a trained
machine learning model and supports **early intervention**.
""")

# ---------------------------------
# Load Models & Features
# ---------------------------------
@st.cache_resource
def load_models():
    features = joblib.load("model_features.pkl")
    
    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
    models = {}
    for name in model_names:
        file_name = name.lower().replace(" ", "_") + "_model.pkl"
        models[name] = joblib.load(file_name)
    
    return features, models

features, models = load_models()

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_excel("stddataset.xlsx")
X = df[features]

default_model = models["Random Forest"]
df["Risk_Probability"] = default_model.predict_proba(X)[:, 1]
df["Predicted_Risk"] = df["Risk_Probability"].apply(
    lambda x: "At Risk" if x >= 0.5 else "Not At Risk"
)

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Overview",
    "🧑‍🎓 Individual Prediction",
    "🧠 Explainable AI"
])

# =====================================================
# TAB 1: DASHBOARD OVERVIEW
# =====================================================
with tab1:
    st.subheader("📊 Risk Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", len(df))
    c2.metric("At Risk Students", (df["Predicted_Risk"] == "At Risk").sum())
    c3.metric("Not At Risk", (df["Predicted_Risk"] == "Not At Risk").sum())

    st.subheader("📈 Risk Distribution")
    fig, ax = plt.subplots()
    df["Predicted_Risk"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("⬇️ Download Prediction Results")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="Student_Risk_Predictions.csv",
        mime="text/csv"
    )

# =====================================================
# TAB 2: INDIVIDUAL STUDENT PREDICTION
# =====================================================
with tab2:
    st.subheader("🧑‍🎓 Individual Student Risk Prediction")

    with st.form("prediction_form"):
        study = st.number_input("Study Hours", 0.0, 24.0, 8.0)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 70.0)
        online = st.number_input("Online Courses", 0, 10, 1)
        assignment = st.number_input("Assignment Completion (%)", 0.0, 100.0, 55.0)
        exam = st.number_input("Exam Score", 0.0, 100.0, 60.0)

        selected_model_name = st.selectbox(
            "Choose Model",
            ["Logistic Regression", "Random Forest", "XGBoost"]
        )
        selected_model = models[selected_model_name]

        submit = st.form_submit_button("Predict Risk")

    if submit:
        input_df = pd.DataFrame(
            [[study, attendance, online, assignment, exam]],
            columns=features
        )

        prob = selected_model.predict_proba(input_df)[0][1]

        if prob >= 0.75:
            level = "🔴 HIGH RISK"
        elif prob >= 0.45:
            level = "🟠 MEDIUM RISK"
        else:
            level = "🟢 LOW RISK"

        st.metric("Risk Probability", f"{prob:.2%}")
        st.subheader(f"Risk Level: {level}")

        st.subheader("📌 Personalized Recommendations")
        recs = []
        if attendance < 75:
            recs.append("• Improve attendance with weekly monitoring.")
        if assignment < 60:
            recs.append("• Provide extra practice assignments.")
        if study < 10:
            recs.append("• Introduce time-management strategies.")
        if exam < 60:
            recs.append("• Arrange remedial classes.")

        if recs:
            for r in recs:
                st.write(r)
        else:
            st.success("Student performance indicators are satisfactory.")

# =====================================================
# TAB 3: EXPLAINABLE AI + CHATBOT
# =====================================================
with tab3:
    st.subheader("🧠 Feature Importance (Explainable AI)")

    selected_model_name_fi = st.selectbox(
        "Choose Model for Feature Importance",
        ["Random Forest", "XGBoost"]
    )
    fi_model = models[selected_model_name_fi]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": fi_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # ---------------- CHATBOT ----------------
    st.divider()
    st.subheader("🤖 Explainable AI Chatbot")

    user_query = st.text_input(
        "Ask why a student is at risk or how to reduce risk:"
    )

    if user_query:
        q = user_query.lower()

        if "attendance" in q:
            st.write("📌 Attendance is the strongest predictor. Low attendance significantly increases risk.")
        elif "assignment" in q:
            st.write("📌 Poor assignment completion indicates low engagement and increases risk.")
        elif "study hours" in q or "study" in q:
            st.write("📌 Students with fewer study hours are more likely to be at risk.")
        elif "exam" in q:
            st.write("📌 Exam scores matter, but consistent effort matters more than one exam.")
        elif "high risk" in q:
            st.write("🔴 High risk usually occurs when attendance, study hours, or assignments fall below thresholds.")
        elif "medium risk" in q:
            st.write("🟠 Medium risk means the student is borderline and needs monitoring.")
        elif "low risk" in q:
            st.write("🟢 Low risk students show strong academic engagement.")
        elif "reduce" in q or "improve" in q:
            st.write("✅ Improve attendance, complete assignments, increase study hours, and attend support classes.")
        elif "best model" in q:
            st.write("🏆 XGBoost usually gives the most accurate predictions, followed by Random Forest.")
        else:
            st.write("🤖 I can explain risk levels, features, and improvement strategies. Try asking about attendance or assignments.")
