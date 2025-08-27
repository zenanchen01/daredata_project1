import streamlit as st, requests, os

API = os.getenv("API_URL", "http://localhost:8000")

st.title("Bank Marketing – Subscription Prediction")

with st.form("bank_form"):
    age = st.number_input("age", min_value=18, max_value=99, value=39)
    job = st.selectbox("job", ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"])
    marital = st.selectbox("marital", ["single","married","divorced","unknown"])
    education = st.selectbox("education", ["primary","secondary","tertiary","unknown"])
    default = st.selectbox("default", ["no","yes","unknown"])
    balance = st.number_input("balance", value=1000)
    housing = st.selectbox("housing", ["no","yes","unknown"])
    loan = st.selectbox("loan", ["no","yes","unknown"])
    contact = st.selectbox("contact", ["cellular","telephone","unknown"])
    day = st.number_input("day", min_value=1, max_value=31, value=15)
    month = st.selectbox("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    campaign = st.number_input("campaign", min_value=1, value=1)
    pdays = st.number_input("pdays", value=999)
    previous = st.number_input("previous", min_value=0, value=0)
    poutcome = st.selectbox("poutcome", ["success","failure","other","unknown"])
    go = st.form_submit_button("Predict")

if go:
    payload = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": int(balance), "housing": housing,
        "loan": loan, "contact": contact, "day": int(day), "month": month,
        "campaign": int(campaign), "pdays": int(pdays),
        "previous": int(previous), "poutcome": poutcome
    }
    r = requests.post(f"{API}/predict", json=payload, timeout=15)
    st.write(r.json())