from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_docs(folder:str = "data")-> list[Document]:
    docs =[]
    for path in Path(folder).iterdir():
        if path.suffix in {".md", ".txt"}:
            docs.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
    return docs

docs = load_docs()
print(f"Loaded {len(docs)} files")

splitter  = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
store = InMemoryVectorStore(embeddings)
store.add_documents(chunks)
print("Indexed. Ask questions about your files.")

while True:
    question = input("\nQuestion (or 'exit'): ")
    if question.strip().lower() in {"exit", "quit"}:
        break
    for i, doc in enumerate(store.similarity_search(question, k=3), start=1):
        print(f"\n#{i} from {doc.metadata['source']}")
        print(doc.page_content[:300].replace("\n", " "))









