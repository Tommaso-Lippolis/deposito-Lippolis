import streamlit as st
from qa_bot import VectorStore, QaBot, RetrieverSettings

st.session_state.vector_store = VectorStore()
st.session_state.qa_bot = QaBot()

# Titolo principale
st.title("Chatbot con RAG 🤖")

# Sidebar per il caricamento dei file
st.sidebar.title("Caricamento Documenti 📂")
uploaded_file = st.sidebar.file_uploader(
    "Carica un documento", 
    type=["pdf", "docx", "txt", "csv", "md"], 
    accept_multiple_files=True
)
if uploaded_file:
    for file in uploaded_file:
        st.write(f"Caricamento del file: {file.type}")
        st.session_state.vector_store.load_document(file)
        st.write("File caricato:", file.name)

    st.button("Crea Vector Store", on_click=lambda: st.session_state.vector_store.build_vector_store())

# Tabs for hyperparameter tuning and chatbot
tabs = st.tabs(["💬 Chatbot", "🔧 Hyperparameter Tuning"])

# Tab 1: Hyperparameter Tuning
with tabs[0]:
    st.header("Chatbot")
    st.write("Interagisci con il chatbot utilizzando il modello configurato.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for chatbot
    if prompt := st.chat_input("Chiedimi qualcosa!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from the chatbot
        with st.chat_message("assistant"):
            response = st.session_state.qa_bot.ask(
                prompt, 
                st.session_state.vector_store.get_retriever(
                    st.session_state.hyperparameters
                )
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Tab 2: Chatbot
with tabs[1]:
    st.header("Hyperparameter Tuning")
    st.write("Modifica i parametri di ricerca per ottimizzare i risultati.")

    # Example hyperparameters
    search_type = st.selectbox("Search Type", options=["similarity", "mmr"], index=0)
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=300, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=100, step=10)
    k = st.slider("Top-K Results", min_value=1, max_value=20, value=4)
    mmr_lambda = st.slider("MMR Lambda", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    # Save hyperparameters in session state
    if st.button("Aggiorna Parametri"):
        st.session_state.hyperparameters = RetrieverSettings(
            search_type=search_type,
            # chunk_size=chunk_size,
            # chunk_overlap=chunk_overlap,
            k=k,
            mmr_lambda=mmr_lambda
        )

        st.success("Parametri aggiornati!")