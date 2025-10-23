import streamlit as st
from back import uc41_back, connection
from css.uc31html import css, bot_template, user_template
from dotenv import load_dotenv, find_dotenv
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import Chroma, FAISS

AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"
path = "/domino/datasets/local/GEN_AI/"
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# def tagging chain
from langchain.chains import create_tagging_chain

schema = {
    "properties": {
        "language": {
            "type": "string",
            "enum": ["english", "french"],
            "description": "The language of the document",
        },
        "country": {
            "type": "string",
            "description": "The country in which the product was manufactured",
        },
        "Stream": {
            "type": "string",
            "enum": ["Technology", "Professional Services"],
            "description": "The type of the spend"
        },
        "price": {
            "type": "integer",
            "description": "The amount in €, if mentioned"
        }
    },
    "required": ["language", "type", "price"],
}



prompt = PromptTemplate.from_template(

"""
You are a professional, polite, and helpful assistant specialized in procurement and financial performance, working for a large banking group.
 
You assist employees by analyzing purchasing data and financial performance indicators. You study perimeters such as family, activity or country, and always rely strictly on the provided context.
 
--- Start of conversation behavior ---
If the user greets you (e.g., "Hello", "Hi", "Bonjour"), respond with a warm and professional welcome, introduce your purpose (assisting in procurement and performance), and suggest what type of help you can offer (e.g., analyzing spend, comparing data, extracting insights).
--- End of conversation behavior ---
If the user says goodbye or thanks you to end the conversation, respond politely and offer assistance again if needed in the future.
 
--- General behavior ---
Always respond using only the information contained in the context below. Never use external or prior knowledge.
Take into account all metadata provided and analyze them separately when relevant (e.g., compare performance between two countries or business lines, but present each with the same level of detail).
If multiple interpretations are possible, ask a clarifying question before responding.
Keep your tone professional, clear, and concise. Refrain from using tables at all time especially for direct questions on spend amounts or savingsamount for a perimeter. Use tables or bullet points only when it improves readability. 
 
if you are asked about vendor or suppliers analysis take into account only informations from sources ending with xlsx (Excel) and follow the rules : 
   - provide amounts for suppliers in M€ or K€ if asked.
   - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.
 
if you are asked about external spend or procurement performance take into account only informations from sources ending with pdf and follow the rules : 
   - provide only the kpi asked.
   - if asked about spend, it is the same as external expenditure.
   - if you are asked about spend amounts, do not give the details by family or by activity only if asked.
   - if you are asked about spend by family or by activity or by country, provide an organized table with amounts.
   - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.
 
 
if you are asked about Savings and the different types (P&L improvement, cost avoidance and savings on projects) take into account only informations from sources ending with pdf and follow the rules : 
   - provide only the kpi asked.
   - if you are asked about savings amounts, do not give the details by family or by activity only if asked.
   - if you are asked about savings by family or by activity or by country, provide an organized table with amounts.
   - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.
 
 
Your source of information is:
---------------------
{context}
---------------------
 
Use the chat history and the follow-up question to generate a self-contained, standalone question that you will answer. 
 
Chat History: {chat_history}
Query: {question} 
 
Answer:
"""
 
)
 
##################################################



def handle_user_input(user_question):

    response=st.session_state.conversation({'question':user_question})

    #st.write(response)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):

        if i % 2==0:

            #st.write(message)

            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        else:

            #st.write(message)

            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def app():

    load_dotenv(find_dotenv())
    llm = connection.create_llm_chat_langchain("gpt4o","2024-02-15-preview", 0.6)
    embeddings = connection.create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL, AZURE_AOAI_API_VERSION)
    persist_directory = "/mnt/Use_cases/Translate_contract/DB/"
    
    vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    )
    #st.set_page_config("Chat with PDF documents", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:

        st.session_state.chat_history=None
    
    st.header("P&P Chatbot 💬")
    st.subheader("Your are reading Spend and Procurement Savings Reports")

    #pdf_docs = st.file_uploader("Ask your question and Click on Process", accept_multiple_files=True)

    st.markdown("""
            <style>
            .main {
                padding: 20px;
                border-radius: 10px;
            }
            .stButton>button {
                ackground-color: #0000FF;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            .stTextInput>div>div>input {
                border: 2px solid #FF69B4;
                border-radius: 8px;
            }
            </style>
        """, unsafe_allow_html=True)
            
    if st.button('Process'):

            with st.spinner("Processing"):
                st.write('it is uploading ')
            result = vectordb.get()
            st.write(f"This DB contains {len(result['metadatas'])} documents")    
            #st.write("data ", text_chunks)
            st.success("Done!")
            

    user_question = st.text_input("Ask a question About Spend or Savings")

    

    if user_question:

        #st.write(user_question)
            # Create Conversation Chain

        st.session_state.conversation=uc41_back.get_conversation_chain(vectordb, llm, embeddings, prompt, user_question)
        handle_user_input(user_question)
