from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

texts = [
    "React hooks like useState let components keep track of data.",
    "The Eiffel Tower is located in Paris, France.",
    "Python uses indentation to define code blocks.",
    "Cats sleep for up to sixteen hours a day.",
]

store = InMemoryVectorStore(embeddings)
store.add_texts(texts)

question = "How do I manage state in a frontend app?"
results = store.similarity_search(question)

print("\n Question", question)

for r in results:
    print("-", r.page_content)




