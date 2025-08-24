import streamlit as st

# aggiungi una emoji in entrambi
st.title("Chatta con GPT-4.1! 🤖")
st.write("Benvenuto nella Home Page! 😊")

st.write("Fai il login per accedere al chatbot.")
if st.button("Vai al Login"):
        st.switch_page("pages/Login.py")