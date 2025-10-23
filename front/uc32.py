import streamlit as st
from back import uc32_back, connection
from dotenv import load_dotenv, find_dotenv
import os
import time
#from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_core.prompts import PromptTemplate


AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"

def app():
    load_dotenv(find_dotenv())
    
    st.title("🈶 Document Translation with GPT")

    file = st.file_uploader("Upload file to be summarized, only [ONE] file allowed", type=["pdf","pptx","docx", "txt"])

    st.subheader("Please specify your translation criteria", anchor=False)
    st.warning("Specify the input language for better perfomance as GPT model may face language recongnition issues.")
    col1, col2, col3 = st.columns(3)
    
    languages_list = os.environ["LANGUAGE_LIST"].split(',')

    with col1:
        input_language = st.selectbox(label="Input Language", options=languages_list)
    with col2:
        output_language = st.selectbox(label="Summary Language", options=languages_list)
    with col3: 
        input_model= st.selectbox(label="Select Model", options=['gpt4o','gpt35turbo', 'mistral-large-2407'])
    

    if input_model == 'mistral-large-2407':
        llm = connection.create_mistral("mistral-large-2407")    
    else : 
        llm = connection.create_llm_chat_langchain(input_model,AZURE_AOAI_API_VERSION)

    if input_model == "gpt4o":
        price = 0.005
    else:
        price = 0.0005
    

    if st.button('Process'):

            with st.spinner("Processing"):
                
                start_time = time.time()
                origninal_text = uc32_back.load_document(file)
                #number_charater = len(origninal_text.split())
                #number_token = llm.get_num_tokens(origninal_text)
                
                with st.expander("See Translated Result"):
                    
                    #output = result
                    output = uc32_back.translate_text_file(llm, file, input_language, output_language)

                    original_text = uc32_back.load_document(file)
                    col3, col4,col5, col6 = st.columns(4)
                    #col3.metric("N° Words(Input)", str(number_charater), "")
                    #col4.metric("N° Token(Input)", str(number_token), "")
                    col5.metric("Price per 1000 token", "$"+str(price), "")
                    #col6.metric("Total Price", "$"+str(price*(number_token/1000)), "")

                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    st.subheader("Origninal_text")
                    st.warning(original_text)
                    st.subheader("Translated_text")
                    st.success(output)
                    st.info(f"This operation has token {time_elapsed:.2f} seconds")
                    
                    
    
