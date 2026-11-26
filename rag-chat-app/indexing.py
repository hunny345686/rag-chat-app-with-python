from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_docs(folder: str = "data") -> list[Document]:
    docs = []
    for path in Path(folder).iterdir():
        if path.suffix in {".md", ".txt"}:
            docs.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
    return docs


def build_store(folder: str = "data") -> InMemoryVectorStore:
    docs = load_docs(folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    print(f"Indexed {len(docs)} files as {len(chunks)} chunks")
    return store
