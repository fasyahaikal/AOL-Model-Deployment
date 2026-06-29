import streamlit as st
import pandas as pd
import pickle

#  Load model & encoder 
model = pickle.load(open("credit_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
target_encoder = pickle.load(open("target_encoder.pkl", "rb"))

#  UI 
st.title("Credit Score Prediction")
st.write("Masukkan data customer untuk memprediksi credit score.")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Month", encoders["Month"].classes_)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
    annual_income = st.number_input("Annual Income", value=50000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary", value=4000.0)
    bank_account = st.number_input("Number of Bank Accounts", min_value=0, value=3)
    credit_card = st.number_input("Number of Credit Cards", min_value=0, value=2)
    interest = st.number_input("Interest Rate", min_value=0, value=5)
    loan = st.number_input("Number of Loans", min_value=0, value=1)
    delay = st.number_input("Delay From Due Date", min_value=0, value=5)
    delayed_payment = st.number_input("Number of Delayed Payments", value=2.0)

with col2:
    changed_limit = st.number_input("Changed Credit Limit", value=10.0)
    credit_inquiries = st.number_input("Number of Credit Inquiries", value=2.0)
    credit_mix = st.selectbox("Credit Mix", encoders["Credit_Mix"].classes_)
    debt = st.number_input("Outstanding Debt", value=500.0)
    utilization = st.number_input("Credit Utilization Ratio", value=30.0)
    history = st.number_input("Credit History Age (months)", value=120.0)
    payment_min = st.selectbox("Payment of Minimum Amount", encoders["Payment_of_Min_Amount"].classes_)
    emi = st.number_input("Total EMI per Month", value=150.0)
    invest = st.number_input("Amount Invested Monthly", value=200.0)
    payment_behaviour = st.selectbox("Payment Behaviour", encoders["Payment_Behaviour"].classes_)
    balance = st.number_input("Monthly Balance", value=500.0)

# Prediksi 
if st.button("Predict Credit Score", use_container_width=True):
    input_data = pd.DataFrame({
        "Month":                    [encoders["Month"].transform([month])[0]],
        "Age":                      [age],
        "Occupation":               [encoders["Occupation"].transform([occupation])[0]],
        "Annual_Income":            [annual_income],
        "Monthly_Inhand_Salary":    [monthly_salary],
        "Num_Bank_Accounts":        [bank_account],
        "Num_Credit_Card":          [credit_card],
        "Interest_Rate":            [interest],
        "Num_of_Loan":              [loan],
        "Delay_from_due_date":      [delay],
        "Num_of_Delayed_Payment":   [delayed_payment],
        "Changed_Credit_Limit":     [changed_limit],
        "Num_Credit_Inquiries":     [credit_inquiries],
        "Credit_Mix":               [encoders["Credit_Mix"].transform([credit_mix])[0]],
        "Outstanding_Debt":         [debt],
        "Credit_Utilization_Ratio": [utilization],
        "Credit_History_Age":       [history],
        "Payment_of_Min_Amount":    [encoders["Payment_of_Min_Amount"].transform([payment_min])[0]],
        "Total_EMI_per_month":      [emi],
        "Amount_invested_monthly":  [invest],
        "Payment_Behaviour":        [encoders["Payment_Behaviour"].transform([payment_behaviour])[0]],
        "Monthly_Balance":          [balance],
    })

    prediction = model.predict(input_data)
    result = target_encoder.inverse_transform(prediction)[0]

    # tampilkan hasil dengan warna sesuai kelas
    if result == "Good":
        st.success(f" Predicted Credit Score: **{result}**")
    elif result == "Standard":
        st.warning(f" Predicted Credit Score: **{result}**")
    else:
        st.error(f" Predicted Credit Score: **{result}**")