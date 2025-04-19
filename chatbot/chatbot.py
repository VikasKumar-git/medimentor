from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import os
import json

# Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# Custom LLM Wrapper
class OllamaLLM:
    def __init__(self, model="llama3.2:latest"):
        self.llm = Ollama(model=model)

    def invoke(self, prompt):
        return self.llm.invoke(prompt)

# Custom Prompt
def set_custom_prompt():
    prompt_template = """
    You are MediMentor, an AI healthcare assistant designed to help users understand non-communicable diseases (NCDs) such as diabetes, hypertension, heart disease, and obesity.
    Your responses should be easy to understand, supportive, and based strictly on the data provided in the context below.

    Context: {context}
    Question: {question}

    Helpful Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA Chain Setup
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm.llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

# Final result logic
def final_result(query):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)

        llm = OllamaLLM()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)

        result = qa.invoke({"query": query})

        if not result.get("result") or len(result["result"].strip()) < 20:
            fallback_response = llm.invoke(query)
            return f"(📄 No useful match found in PDFs)\n{fallback_response}"

        return result["result"].strip()

    except Exception as e:
        return f"MediMentor: Sorry! I encountered an error: {e}"

# Command line chatbot loop
if __name__ == "__main__":
    print("MediMentor:\n")
    while True:
        user_input = input("YOU: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = final_result(user_input)
        print(f"MediMentor: {response}\n")
