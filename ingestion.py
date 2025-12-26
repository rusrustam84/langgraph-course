from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def digest() -> VectorStoreRetriever:
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)

    doc_splitter = text_splitter.split_documents(docs_list)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=False,
        chunk_size=50,
        retry_min_seconds=10,
    )

    vectorstore = PineconeVectorStore.from_documents(doc_splitter, index_name="langchain-rag-index",embedding=embeddings)

    return vectorstore.as_retriever()

def get_retriever():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=False,
        chunk_size=50,
        retry_min_seconds=10,
    )

    return PineconeVectorStore(index_name="langchain-rag-index", embedding=embeddings).as_retriever()



if __name__ == "__main__":
    # digest()
    get_retriever()