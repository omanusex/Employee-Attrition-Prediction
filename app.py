# Streamlit app to predict employee attrition
# Run with:  streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👔", layout="wide")

MODEL_PATH = "model/attrition_model.pkl"
RESULTS_PATH = "model/results.csv"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run the notebook (Employee_Attrition_Prediction.ipynb) first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


saved = load_model()
model = saved["model"]
scaler = saved["scaler"]
columns = saved["columns"]
cap_limits = saved["cap_limits"]
multi_cat_cols = saved["multi_cat_cols"]
defaults = saved["defaults"]
categories = saved["categories"]


def prepare_input(employee):
    """Apply the same preprocessing steps used in the notebook to one employee."""
    data = pd.DataFrame([employee])

    # 1) capping (same limits as training)
    for col, (low, high) in cap_limits.items():
        data[col] = data[col].clip(low, high)

    # 2) label encoding
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

    # 3) one-hot encoding, then make sure we have the same columns as the training data
    data = pd.get_dummies(data, columns=multi_cat_cols, dtype=int)
    data = data.reindex(columns=columns, fill_value=0)

    # 4) scaling
    return pd.DataFrame(scaler.transform(data), columns=columns)


st.title("👔 Employee Attrition Prediction")
st.write("This app predicts if an employee is likely to **leave the company**, "
         "using a model trained on the IBM HR Analytics dataset.")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Results"])

# ------------------------------------------------------------------ prediction tab
with tab1:
    st.subheader("Enter the employee details")
    st.caption("Fields that are not shown use the most common value in the dataset.")

    employee = dict(defaults)  # start from the default values

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal**")
        employee["Age"] = st.slider("Age", 18, 60, int(defaults["Age"]))
        employee["Gender"] = st.selectbox("Gender", categories["Gender"])
        employee["MaritalStatus"] = st.selectbox("Marital Status", categories["MaritalStatus"],
                                                 index=categories["MaritalStatus"].index(defaults["MaritalStatus"]))
        employee["DistanceFromHome"] = st.slider("Distance From Home (km)", 1, 29, int(defaults["DistanceFromHome"]))
        employee["Education"] = st.selectbox("Education (1=Below College ... 5=Doctor)", [1, 2, 3, 4, 5], index=2)

    with col2:
        st.markdown("**Job**")
        employee["Department"] = st.selectbox("Department", categories["Department"],
                                              index=categories["Department"].index(defaults["Department"]))
        employee["JobRole"] = st.selectbox("Job Role", categories["JobRole"],
                                           index=categories["JobRole"].index(defaults["JobRole"]))
        employee["JobLevel"] = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=int(defaults["JobLevel"]) - 1)
        employee["MonthlyIncome"] = st.number_input("Monthly Income ($)", 1000, 20000, int(defaults["MonthlyIncome"]), step=100)
        employee["OverTime"] = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
        employee["BusinessTravel"] = st.selectbox("Business Travel", categories["BusinessTravel"],
                                                  index=categories["BusinessTravel"].index(defaults["BusinessTravel"]))
        employee["StockOptionLevel"] = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=int(defaults["StockOptionLevel"]))

    with col3:
        st.markdown("**Experience & Satisfaction**")
        employee["TotalWorkingYears"] = st.slider("Total Working Years", 0, 40, int(defaults["TotalWorkingYears"]))
        employee["YearsAtCompany"] = st.slider("Years At Company", 0, 40, int(defaults["YearsAtCompany"]))
        employee["NumCompaniesWorked"] = st.slider("Number of Companies Worked", 0, 9, int(defaults["NumCompaniesWorked"]))
        employee["JobSatisfaction"] = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=int(defaults["JobSatisfaction"]))
        employee["EnvironmentSatisfaction"] = st.select_slider("Environment Satisfaction", [1, 2, 3, 4],
                                                               value=int(defaults["EnvironmentSatisfaction"]))
        employee["WorkLifeBalance"] = st.select_slider("Work Life Balance", [1, 2, 3, 4], value=int(defaults["WorkLifeBalance"]))
        employee["JobInvolvement"] = st.select_slider("Job Involvement", [1, 2, 3, 4], value=int(defaults["JobInvolvement"]))

    if st.button("Predict", type="primary"):
        X = prepare_input(employee)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ This employee is **likely to LEAVE** the company (probability: {probability:.1%})")
        else:
            st.success(f"✅ This employee is **likely to STAY** in the company (probability of leaving: {probability:.1%})")
        st.progress(float(probability))

# ------------------------------------------------------------------ results tab
with tab2:
    st.subheader("Model performance on the test set")
    st.write("Selected model:", f"**{saved['model_name']}**")

    if os.path.exists(RESULTS_PATH):
        results = pd.read_csv(RESULTS_PATH)
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.bar_chart(results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]], stack=False, horizontal=True)

    c1, c2 = st.columns(2)
    if os.path.exists("figures/13_confusion_matrices.png"):
        c1.image("figures/13_confusion_matrices.png", caption="Confusion matrices")
    if os.path.exists("figures/14_roc_curves.png"):
        c2.image("figures/14_roc_curves.png", caption="ROC curves")
    if os.path.exists("figures/11_feature_importance.png"):
        st.image("figures/11_feature_importance.png", caption="Feature importance")
