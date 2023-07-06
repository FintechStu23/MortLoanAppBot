import streamlit as st

def loan_approval_chatbot():
    st.title("Loan Approval Chatbot")
    st.write("Welcome! How can I assist you with your loan approval?")

    name = st.text_input("Please enter your name:")
    gender = st.selectbox("Please select your gender:", ("Male", "Female", "Other"))
    marital_status = st.selectbox("Please select your marital status:", ("Single", "Married"))
    coapp_income = st.number_input("Please enter the co-applicant's monthly income:", value=0)
    self_employed = st.selectbox("Are you self-employed?", ("No", "Yes"))
    credit_history = st.selectbox("Do you have a credit history?", ("No", "Yes"))
    income = st.number_input("Please enter your monthly income:")

    if st.button("Check eligibility"):
        eligibility = False

        if (income > 5000 and credit_history == "Yes" and gender == "Male") or (income > 3000 and credit_history == "Yes" and gender == "Female"):
            eligibility = True

        if eligibility:
            st.success(f"Congratulations, {name}! You are eligible for the loan.")
        else:
            st.error(f"Sorry, {name}. You are not eligible for the loan.")

    st.write("Thank you for using our loan approval chatbot!")

if __name__ == '__main__':
    loan_approval_chatbot()
