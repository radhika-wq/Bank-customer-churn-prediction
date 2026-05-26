import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


MODEL_PATH = "models/churn_model.pkl"
DATA_PATH = "data/bank_churn.csv"


st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.write("Predict whether a bank customer is likely to leave using Machine Learning.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard Overview", "Customer Churn Prediction", "Project Info"]
)


if page == "Dashboard Overview":
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Existing Customers", int((df["Attrition_Flag"] == "Existing Customer").sum()))
    col3.metric("Attrited Customers", int((df["Attrition_Flag"] == "Attrited Customer").sum()))

    fig = px.pie(
        df,
        names="Attrition_Flag",
        title="Customer Churn Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head())


elif page == "Customer Churn Prediction":
    st.subheader("Predict Customer Churn Risk")

    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
    gender = st.selectbox("Gender", ["M", "F"])
    dependent_count = st.number_input("Dependent Count", min_value=0, max_value=10, value=2)
    education_level = st.selectbox(
        "Education Level",
        ["High School", "Graduate", "Uneducated", "Unknown", "College", "Post-Graduate", "Doctorate"]
    )
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"])
    income_category = st.selectbox(
        "Income Category",
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
    )
    card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])

    months_on_book = st.number_input("Months on Book", min_value=1, max_value=100, value=36)
    total_relationship_count = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=4)
    months_inactive_12_mon = st.number_input("Months Inactive in Last 12 Months", min_value=0, max_value=12, value=2)
    contacts_count_12_mon = st.number_input("Contacts Count in Last 12 Months", min_value=0, max_value=10, value=2)

    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0)
    total_revolving_bal = st.number_input("Total Revolving Balance", min_value=0.0, value=1000.0)
    avg_open_to_buy = st.number_input("Average Open To Buy", min_value=0.0, value=4000.0)
    total_amt_chng_q4_q1 = st.number_input("Total Amount Change Q4/Q1", min_value=0.0, value=0.75)
    total_trans_amt = st.number_input("Total Transaction Amount", min_value=0.0, value=4000.0)
    total_trans_ct = st.number_input("Total Transaction Count", min_value=0, value=60)
    total_ct_chng_q4_q1 = st.number_input("Total Count Change Q4/Q1", min_value=0.0, value=0.7)
    avg_utilization_ratio = st.number_input("Average Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)

    if st.button("Predict Churn Risk"):
        input_data = pd.DataFrame([{
            "Customer_Age": customer_age,
            "Gender": gender,
            "Dependent_count": dependent_count,
            "Education_Level": education_level,
            "Marital_Status": marital_status,
            "Income_Category": income_category,
            "Card_Category": card_category,
            "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_relationship_count,
            "Months_Inactive_12_mon": months_inactive_12_mon,
            "Contacts_Count_12_mon": contacts_count_12_mon,
            "Credit_Limit": credit_limit,
            "Total_Revolving_Bal": total_revolving_bal,
            "Avg_Open_To_Buy": avg_open_to_buy,
            "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4_q1,
            "Total_Trans_Amt": total_trans_amt,
            "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4_q1,
            "Avg_Utilization_Ratio": avg_utilization_ratio
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.metric("Churn Probability", f"{probability * 100:.2f}%")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")


else:
    st.subheader("Project Information")

    st.write("""
    This project predicts bank customer churn using machine learning.

    Features:
    - Customer churn prediction
    - Interactive dashboard
    - User-friendly financial input fields
    - Churn probability score
    - Dataset overview and visualizations

    ML concepts used:
    - Classification
    - Categorical encoding
    - Feature scaling
    - ColumnTransformer
    - Scikit-learn Pipeline
    - Random Forest Classifier
    - ROC-AUC evaluation
    """)