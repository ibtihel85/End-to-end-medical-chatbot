from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import requests
from src.prompt import system_prompt
import pinecone
from pinecone import Pinecone


# -----------------------------
# 1. Custom GroqLLM wrapper
# -----------------------------
from langchain.llms.base import LLM

class GroqLLM(LLM):
    """LangChain wrapper for Groq API"""

    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        api_key = os.environ.get("GROK_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(url, headers=headers, json=data).json()
        print("Groq raw response:", response)
        return response["choices"][0]["message"]["content"]

# -----------------------------
# 2. Flask app setup
# -----------------------------
app = Flask(__name__)
app.static_folder = 'static'  
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = "mchatbot"

# -----------------------------
# 3. Load embeddings + Pinecone
# -----------------------------
from src.helper import download_hugging_face_embeddings
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

# Load an existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)

# -----------------------------
# 4. Prompt template
# -----------------------------


PROMPT = PromptTemplate(
    template=system_prompt, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# -----------------------------
# 5. QA Chain with Groq
# -----------------------------
llm = GroqLLM()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

# -----------------------------
# 6. Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print("User:", msg)
        result = qa.invoke({"query": msg})
        print("Retrieved Documents:", result["source_documents"])  # Debug: Print retrieved docs
        print("Bot:", result["result"])
        return str(result["result"])
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong: " + str(e)

# -----------------------------
# 7. Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
