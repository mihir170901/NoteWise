import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read API key from .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# Setup ChromaDB (local persistent storage)
chroma_client = chromadb.PersistentClient(path="./my_second_brain")

# Use OpenAI embeddings for semantic search
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

# Create or load collection
collection = chroma_client.get_or_create_collection(
    name="notes",
    embedding_function=openai_ef
)

# -------------------------------
# Function to add a note
# -------------------------------
def add_note(note: str):
    # Use current count as unique ID
    all_ids = collection.get()["ids"]
    new_id = str(len(all_ids) + 1)

    collection.add(
        documents=[note],
        ids=[new_id]
    )
    print("✅ Note added!")

# -------------------------------
# Function to query notes
# -------------------------------
def query_notes(question: str):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]

    if not docs:
        return "No relevant notes found."

    # Ask LLM to answer based on retrieved docs
    context = "\n".join(docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant with memory."},
            {"role": "user", "content": f"Here are my notes:\n{context}\n\nNow answer this question: {question}"}
        ]
    )
    return response.choices[0].message.content

# -------------------------------
# Function to list all notes
# -------------------------------
def list_notes():
    docs = collection.get()["documents"]
    if not docs:
        print("ℹ️ No notes found.")
        return
    print("\n📄 Your Notes:")
    for i, note in enumerate(docs, start=1):
        print(f"{i}. {note}")


# -------------------------------
# Main program loop
# -------------------------------
if __name__ == "__main__":
    print("🧠 Welcome to your AI Second Brain!")
    print("You can add notes and later ask questions about them.")

    while True:
        mode = input("\nChoose: [1] Add note  [2] Ask question  [3] List notes  [q] Quit\n> ")
        if mode == "1":
            note = input("Enter your note:\n> ")
            add_note(note)
        elif mode == "2":
            q = input("Ask your question:\n> ")
            answer = query_notes(q)
            print("\n🤖", answer)
        elif mode == "3":
            list_notes()
        elif mode.lower() == "q":
            print("👋 Goodbye! Your notes are saved in ./my_second_brain")
            break
        else:
            print("❌ Invalid option. Please choose 1, 2, 3, or q.")