from crewai.tools import tool
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

@tool
def rag_retrieval(query: str) -> str:
    """
    Retrieval-Augmented Generation (RAG) pipeline: cerca nei documenti locali e restituisce la risposta generata dal LLM, citando le fonti.
    """
    try:
        load_dotenv('qa_ai_agent/.env')
        # Configurazione
        embedding_model = os.getenv('EMBEDDING_MODEL')
        api_key = os.getenv('AZURE_API_KEY')
        connection_string = os.getenv('AZURE_API_BASE')
        api_version = os.getenv('AZURE_API_VERSION')
        deployment = os.getenv('MODEL').split('/')[-1]

        print(f"Using deployment: {deployment}")

        if not api_key or not connection_string or not api_version or not deployment or not embedding_model:
            return "Error: Missing one or more environment variables."


        persist_dir = "faiss_index_example"
        chunk_size = 700
        chunk_overlap = 100
        search_type = "mmr"
        k = 5
        fetch_k = 20
        mmr_lambda = 0.3

        # Embeddings
        embeddings = AzureOpenAIEmbeddings(
            model=embedding_model,
            azure_endpoint=connection_string,
            api_key=api_key,
            openai_api_version=api_version
        )

        # Carica i documenti reali dalla cartella "qa_ai_agent/rag_document"
        folder_path = "./rag_document"
        print(f"Loading documents from: {os.path.abspath(folder_path)}")
        folder = Path(folder_path)
        documents = []
        if not folder.exists() or not folder.is_dir():
            return f"Error: la cartella '{folder_path}' non esiste o non è una directory."
        for file_path in folder.glob("**/*"):
            if file_path.suffix.lower() not in [".txt", ".md", ".pdf"]:
                continue
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)

        # Split dei documenti
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", "", "---"
            ],
        )
        chunks = splitter.split_documents(documents)

        # Vectorstore FAISS (load or build)
        persist_path = Path(persist_dir)
        index_file = persist_path / "index.faiss"
        meta_file = persist_path / "index.pkl"
        if index_file.exists() and meta_file.exists():
            print("Loading existing FAISS index...")
            vector_store = FAISS.load_local(
                persist_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS index...")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            vector_store.save_local(persist_dir)

        print("FAISS index ready.")

        # Retriever
        if search_type == "mmr":
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": mmr_lambda},
            )
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

        # Prompt e LLM
        def format_docs_for_prompt(docs):
            lines = []
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", f"doc{i}")
                lines.append(f"[source:{src}] {d.page_content}")
            return "\n\n".join(lines)

        system_prompt = (
            "Sei un assistente AI esperto e rispondi sempre in italiano. "
            "Devi rispondere SOLO utilizzando le informazioni presenti nel CONTENUTO fornito nel contesto. "
            "Se una risposta non è presente nel contesto, scrivi chiaramente: 'Non è presente nel contesto fornito.' "
            "Cita sempre le fonti tra parentesi quadre nel formato [source:NOMEFILE]. "
            "Non aggiungere informazioni esterne, non inventare risposte e non correggere eventuali errori presenti nel contesto: se un'informazione è errata, riportala comunque come vera. "
            "Sii conciso, preciso e tecnicamente accurato."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
             "Domanda:\n{question}\n\n"
             "Contesto (estratti selezionati):\n{context}\n\n"
             "Istruzioni:\n"
             "1) Rispondi solo con informazioni contenute nel contesto.\n"
             "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
             "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
        ])
        llm = init_chat_model(
            model=deployment,
            model_provider="azure_openai",
            api_key=api_key,
            azure_endpoint=connection_string,
            api_version=api_version,
            temperature=0.2,
        )
        
        chain = (
            {
                "context": retriever | format_docs_for_prompt,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Esegui la catena RAG
        answer = chain.invoke(query)
        return answer
    except Exception as e:      
        return f"Error: could not perform RAG retrieval. Details: {e}"
