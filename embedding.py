import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from pymongo import MongoClient
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")


if __name__ == "__main__":

    llm = AzureChatOpenAI(
        openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
        azure_deployment="gpt",
        temperature=0,
    )

    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embedding",
        openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
        chunk_size=10
    )

    loader = PyMuPDFLoader(file_path="./doc/Agency_Authorization_Playbook.pdf")
    docs = loader.load()

    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client['mydatabase']['mycontainer']

    vectorstore = AzureCosmosDBVectorSearch.from_documents(
        docs,
        aoai_embeddings,
        collection=collection,
        index_name='vectorSearchIndex',
    )
    
    num_lists = 100
    dimensions = 1536
    similarity_algorithm = CosmosDBSimilarityType.COS
    kind = CosmosDBVectorSearchType.VECTOR_IVF
    m = 16
    ef_construction = 64
    ef_search = 40
    score_threshold = 0.1
    
    vectorstore.create_index(
        num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
    )

    print('success!')
