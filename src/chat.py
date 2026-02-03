from search import search_prompt
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


# similarity_search_with_score(query, k=10)

load_dotenv()








model = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context. Return 'Não tenho informações necessárias para responder sua pergunta' if a question is out of the context."),
    ("user", "{input}"),
])

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="formal", )

print(message)

# def main():
#     chain = search_prompt()

#     if not chain:
#         print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
#         return
    
#     pass

# if __name__ == "__main__":
#     main()