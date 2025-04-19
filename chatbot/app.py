import requests
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import json
import os

# Define paths
DB_FAISS_PATH = "vectorstores/db_faiss"
USER_DATA_PATH = "userdata/"

# Custom prompt template for MediMentor
custom_prompt_template = """
You are MediMentor, a friendly and helpful assistant that provides guidance on lifestyle disease prevention and management.

User Lifestyle Details:
{user_context}

Instructions:
1. Stay focused on topics like lifestyle, diet, exercise, health risks, disease prevention, and general well-being.
2. Keep responses brief, supportive, and educational.

Context: {context}
Question: {query}
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question', 'user_context'])
    return prompt

# Custom LLM class using Ollama
class OllamaLLM(LLM):
    def _call(self, prompt: str, stop=None):
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3",
            "prompt": prompt,
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=data)
        try:
            responses = response.text.strip().split('\n')
            collected_responses = []
            for response_str in responses:
                parsed_response = json.loads(response_str)
                if parsed_response.get("response"):
                    collected_responses.append(parsed_response["response"])
            return ''.join(collected_responses).strip()
        except ValueError:
            raise Exception(f"Ollama API call failed with invalid JSON. Response: {response.text}")

    @property
    def _identifying_params(self):
        return {"name_of_model": "ollama_custom"}

    @property
    def _llm_type(self):
        return "ollama_custom"

# Validate response based on context
def validate_response(response, context):
    if any(word in response for word in context.split()):
        return response.strip()
    return "I don't know the answer."

# Setup RetrievalQA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def load_user_data(user_name):
    file_path = os.path.join(USER_DATA_PATH, f"{user_name.lower().replace(' ', '_')}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def qa_bot(user_context):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = OllamaLLM()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query, user_context):
    qa_chain = qa_bot(user_context)
    try:
        result = qa_chain.invoke({
            "question": query,
            "user_context": user_context
        })
        answer = result.get("result", "I don't know the answer based on the information I have.")
        context = " ".join([doc.page_content for doc in result.get("source_documents", [])])
        return validate_response(answer, context)
    except Exception as e:
        return f"I encountered an error: {str(e)}"

# Main chat loop
if __name__ == "__main__":
    print("\n💬 Welcome to MediMentor! Please enter your name to get started.")
    user_name = input("Enter your full name: ")
    user_data = load_user_data(user_name)

    if not user_data:
        print("Sorry, we couldn't find your details. Please register first.")
        exit()

    user_context = "\n".join([f"{key}: {value}" for key, value in user_data.items()])
    print("\n✅ Profile loaded! You can now ask anything related to your health and lifestyle. Type 'exit' to quit.\n")

    while True:
        query = input("YOU: ")
        if query.lower() in ["exit", "quit"]:
            print("MediMentor: Take care and stay healthy! 👋")
            break
        try:
            answer = final_result(query, user_context)
            print(f"MediMentor: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
