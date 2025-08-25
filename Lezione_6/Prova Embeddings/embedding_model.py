from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv('Lezione_6/.env')

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
connection_string = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=connection_string
)

# Funzione per ottenere gli embeddings
def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model=deployment 
    )
    return response.data[0].embedding

if __name__ == "__main__":
    text = "Hello World"

    print(f"Testo: {text}")

    embedding = get_embeddings(text)

    print(f"Dimensione embedding del modello: {len(embedding)}")
    print(f"Embedding: {embedding[:5]}")

    # generiamo due esempi di embeddings
    text1 = "Gli animali sono esseri viventi che appartengono al regno animale."
    text2 = "Il gatto è un animale domestico."

    print(f"Testo 1: {text1}")
    print(f"Testo 2: {text2}")

    embedding1 = get_embeddings(text1)
    embedding2 = get_embeddings(text2)

    print(f"Embedding 1: {embedding1[:5]}")
    print(f"Embedding 2: {embedding2[:5]}")

    # Calcolo della similarità coseno
    similarity = cosine_similarity([embedding1], [embedding2])

    print(f"Cosine Similarity: {similarity[0][0]}")


