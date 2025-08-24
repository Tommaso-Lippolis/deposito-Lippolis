from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv("Lezione_5\.env")

if "authenticated" not in st.session_state:
    st.error("Devi prima autenticarti!")
    if st.button("Vai al Login"):
        st.switch_page("pages/Login.py")
    # st.markdown("[Vai al Login](login)")  se si vuole un link che apra la pagina di login in una nuova scheda

else:
    client = AzureOpenAI(
        api_version=st.session_state.api_version,
        api_key=st.session_state.api_key,
        azure_endpoint=st.session_state.endpoint
    )

    st.title("Chat con GPT-4.1")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Chiedimi qualcosa! 😊"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state.deployment,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


