import streamlit as st
from back import uc33_back, connection
from dotenv import load_dotenv, find_dotenv
import os
import time
import numpy as np
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
#from langchain_mistralai import ChatMistralAI


AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"
MISTRAL_MEDIUM = "mistral-medium-4bit-GPTQ"
MISTRAL_LARGE = "mistral-large-2407"

def app():
    load_dotenv(find_dotenv())

    if 'pdf_ref' not in ss:
        ss.pdf_ref = None

    st.title("Document Summarization")


    file = st.file_uploader("Upload file to be summarized, only [ONE] file allowed", type=["pdf","pptx","docx", "txt"],key='pdf')

    tab1, tab2, tab3 = st.tabs(["Origin","Summary","ChatBox"])


    with tab1:

        if file:

            if ss.pdf:
                ss.pdf_ref = ss.pdf 

            if ss.pdf_ref:

                binary_data = ss.pdf_ref.getvalue()
                pdf_viewer(input=binary_data, width=1000)

    with tab2:
        st.subheader("Please specify your summary criteria", anchor=False)
        st.warning("Specify also the input language for better perfomance as GPT model may face language recongnition issues.")


        col3, col4 = st.columns(2)
        languages_list = os.environ["LANGUAGE_LIST"].split(',')

        with col3:
            input_language = st.selectbox(label="Input Language", options=languages_list)
        with col4:
            summary_language = st.selectbox(label="Output Language", options=languages_list)
            

        col5, col6 = st.columns(2)
        with col5:
            input_model= st.selectbox(label="Select Model", options=['gpt4o','gpt35turbo', "mistral-large"])
            
        with col6:
            summarize_methode= st.selectbox(label="Summary Mesthode", options=['map_reduce', 'stuff', 'refine'])

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Spontaneity Level", 0.0, 1.0, 0.0, 0.1)
        with col2:
            sum_len = st.slider("Maximum Output Length", 0, 1000, 50, 50)

        #llm = connection.create_llm_chat_langchain(input_model,AZURE_AOAI_API_VERSION, temperature)
        if input_model == 'mistral-large-2407':
            llm = connection.create_mistral("mistral-large-2407")
        else : 
            llm = connection.create_llm_chat_langchain(input_model,AZURE_AOAI_API_VERSION, temperature)

        
        if input_model == "gpt4o":
            price = 0.005
        else:
            price = 0.0005

        if st.button('Process'):

            with st.spinner("Processing"):
                start_time = time.time()
                input_text = uc33_back.load_document(file)
                input_word_nb = len(input_text.split())
                input_token_nb = llm.get_num_tokens(input_text)
                token_rate = input_token_nb/input_word_nb
                output_token_nb = int(np.floor(sum_len*token_rate))
                col7, col8, col9, col10, col11, col12 = st.columns(6)
                col7.metric("N° Words [Input]", str(input_word_nb), "")
                #col8.metric("N° Token [Input]", str(input_token_nb), "")
                #col9.metric("N° Words [Output]", str(sum_len), "")
                #col10.metric("N° Token [Output]", str(output_token_nb), "")
                col11.metric("Price per 1000 token", "$"+str(price), "")
                #col12.metric("Total Price", "$"+str(price*((input_token_nb+output_token_nb)/1000)), "")

                new_summary = uc33_back.get_summary(llm,file,summarize_methode,sum_len,input_language, summary_language)

                with st.expander("See Summarized Result"):
                    st.info(new_summary)
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    st.info(f"This operation has token {time_elapsed:.2f} seconds")

    with tab3:
        st.header("ChatBox")
        st.write("🤖 Welcome to the AI Assistant ChatBox!") 
        st.write("Got burning questions about your data science problem or need help navigating the intricacies of your project? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

        input_question = st.text_area(label="Tap your Question here", key="input_question")
            
        col5, col6 = st.columns(2)
        with col5:
            input_model= st.selectbox(label="Select Model", options=['gpt4o','gpt35turbo', "mistral-large"], key="input_model_chat")
        if input_model == 'mistral-large-2407':
            llm = connection.create_mistral("mistral-large-2407")
        else : 
            llm = connection.create_llm_chat_langchain(input_model,AZURE_AOAI_API_VERSION, temperature)

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Spontaneity Level", 0.0, 1.0, 0.0, 0.1, key="temperature_chat")
        with col2:
            sum_len = st.slider("Maximum Output Length", 0, 1000, 50, 50, key="sum_len_chat")
        
        if input_model == "gpt4o":
            price = 0.005
        else:
            price = 0.0005

        if st.button('Process', key="process_button_chat"):

            with st.spinner("Processing"):
                start_time = time.time()
                input_text = uc33_back.load_document(file)
                input_word_nb = len(input_text.split())
                #input_token_nb = llm.get_num_tokens(input_text)
                #token_rate = input_token_nb/input_word_nb
                #output_token_nb = int(np.floor(sum_len*token_rate))
                col7, col8, col9, col10, col11, col12 = st.columns(6)
                col7.metric("N° Words [Input]", str(input_word_nb), "")
                #col8.metric("N° Token [Input]", str(input_token_nb), "")
                #col9.metric("N° Words [Output]", str(sum_len), "")
                #col10.metric("N° Token [Output]", str(output_token_nb), "")
                col11.metric("Price per 1000 token", "$"+str(price), "")
                #col12.metric("Total Price", "$"+str(price*((input_token_nb+output_token_nb)/1000)), "")

                Answer = uc33_back.chat_pdf(llm,file,sum_len,input_question)

                with st.expander("See The Answer"):
                    st.info(Answer)
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    st.info(f"This operation has token {time_elapsed:.2f} seconds")

