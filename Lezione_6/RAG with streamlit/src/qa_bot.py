import os
import tempfile
from typing import List, Optional
from attr import dataclass

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

from langchain_azure import AzureChatModel

load_dotenv("Lezione_6/RAG with streamlit/.env")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM")


@dataclass
class RetrieverSettings:
    search_type: str = "similarity" # "mmr" o "similarity"
    k: int = 4
    fetch_k: int = 10
    mmr_lambda: float = 0.5


class VectorStore:
    def __init__(
        self, 
        embedding_model: str = "text-embedding-ada-002",
        vector_store_type: str = "faiss",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the QaBot with specified configurations.
        
        Args:
            embedding_model: The embedding model to use
            vector_store_type: Type of vector store ("faiss")
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        Settings = RetrieverSettings()
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=embedding_model,
            azure_endpoint=endpoint,
            api_key=azure_api_key,
            )
        # if "openai" in embedding_model.lower():
        #     self.embeddings = OpenAIEmbeddings(model=embedding_model)
        # else:
        #     self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self.vector_store = None
        self.documents = []

    def load_document(self, uploaded_file):
        """
        Load a document uploaded by Streamlit's file uploader.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
        """
        # if not isinstance(uploaded_file, st.uploaded_file_manager.UploadedFile):
        #     raise ValueError("Invalid uploaded file")
        
        # Get file extension
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        try:
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            # Load document based on file type
            if file_extension in ['.txt', '.md']:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(tmp_file_path)
            else:
                raise ValueError("Unsupported file type")
            
            # Load the document
            docs = loader.load()
            
            # Add metadata about the source
            for doc in docs:
                doc.metadata['source'] = file_name
                doc.metadata['uploaded'] = True

            self.documents.extend(docs)
            print(f"Loaded document: {file_name} ({len(docs)} pages/sections)")
            
        except Exception as e:
            raise e
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    def load_multiple_documents(self, uploaded_files: list):
        """
        Load multiple documents uploaded by Streamlit's file uploader.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
        """
        
        for uploaded_file in uploaded_files:
            self.load_document(uploaded_file)

        print(f"Total documents loaded from uploads: {len(self.documents)}")

        return self.documents

    # def load_documents_from_path(self, path: str, file_types: List[str] = None) -> List[Document]:
    #     """
    #     Load documents from a given path.
        
    #     Args:
    #         path: Path to the directory containing documents
    #         file_types: List of file extensions to load (e.g., ['.txt', '.pdf', '.csv'])
        
    #     Returns:
    #         List of loaded documents
    #     """
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"Path {path} does not exist")
        
    #     if file_types is None:
    #         file_types = ['.txt', '.pdf', '.csv', '.md']
        
    #     documents = []
        
    #     # Load different file types
    #     for file_type in file_types:
    #         try:
    #             if file_type == '.txt' or file_type == '.md':
    #                 loader = DirectoryLoader(
    #                     path, 
    #                     glob=f"**/*{file_type}",
    #                     loader_cls=TextLoader,
    #                     loader_kwargs={'encoding': 'utf-8'}
    #                 )
    #             elif file_type == '.pdf':
    #                 loader = DirectoryLoader(
    #                     path, 
    #                     glob=f"**/*{file_type}",
    #                     loader_cls=PyPDFLoader
    #                 )
    #             elif file_type == '.csv':
    #                 loader = DirectoryLoader(
    #                     path, 
    #                     glob=f"**/*{file_type}",
    #                     loader_cls=CSVLoader
    #                 )
    #             else:
    #                 continue
                
    #             docs = loader.load()
    #             documents.extend(docs)
    #             print(f"Loaded {len(docs)} {file_type} documents")
                
    #         except Exception as e:
    #             print(f"Error loading {file_type} files: {str(e)}")
        
    #     self.documents = documents
    #     print(f"Total documents loaded: {len(documents)}")
    #     return documents
    
    def build_vector_store(self) -> None:
        """
        Build a vector store from the loaded documents.
        
        Args:
            documents: Optional list of documents. If None, uses self.documents
        """
        if not self.documents:
            raise ValueError("No documents provided. Load documents first using load_documents()")
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        doc_chunks = self.text_splitter.split_documents(self.documents)
        print(f"Created {len(doc_chunks)} document chunks")
        
        # Build vector store
        print(f"Building {self.vector_store_type} vector store...")
        
        if self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=doc_chunks,
                embedding=self.embeddings
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        print("Vector store built successfully!")
    
    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path where to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Build vector store first.")
        
        if self.vector_store_type.lower() == "faiss":
            self.vector_store.save_local(path)
        else:
            print("Save functionality not implemented for this vector store type")
        
        print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str) -> None:
        """
        Load a previously saved vector store from disk.
        
        Args:
            path: Path where the vector store is saved
        """
        if self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.load_local(
                path, 
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Load functionality not implemented for this vector store type")
        
        print(f"Vector store loaded from {path}")
    
    def get_retriever(self, settings: RetrieverSettings):
        if settings.search_type == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": settings.k, 
                    "fetch_k": settings.fetch_k, 
                    "lambda_mult": settings.mmr_lambda
                },
            )
        else:
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.k},
            )
        
    def get_stats(self) -> dict:
        """
        Get statistics about the loaded documents and vector store.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store_built": self.vector_store is not None
        }
        
        if self.vector_store and hasattr(self.vector_store, 'index'):
            if self.vector_store_type.lower() == "faiss":
                stats["vector_count"] = self.vector_store.index.ntotal
        
        return stats

def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

class QaBot:

    def __init__(self):
        self.llm_client = AzureChatModel()

    def ask(self, question: str, retriever) -> str:
        system_prompt = (
            "Sei un assistente esperto. Rispondi in italiano. "
            "Usa esclusivamente il CONTENUTO fornito nel contesto. "
            "Se l'informazione non è presente, dichiara che non è disponibile. "
            "Includi citazioni tra parentesi quadre nel formato [source:...]. "
            "Sii conciso, accurato e tecnicamente corretto."
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

        chain = (
            {
                "context": retriever | format_docs_for_prompt,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_client
            | StrOutputParser()
        )
        response = chain.invoke(question)
        return response