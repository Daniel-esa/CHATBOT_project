import streamlit as st
from back import uc31_back, connection
from css.uc31html import css, bot_template, user_template
from dotenv import load_dotenv, find_dotenv


from langchain_chroma import Chroma

AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt-4o"
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt-4o-mini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada-002"

def handle_user_input(user_question):

    response=st.session_state.conversation.invoke({'question':user_question})

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
    llm = connection.create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O, 0.6)
    embeddings = connection.create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL)
    persist_directory = "./dbs/chroma/temp/uc31/test"
    
    #st.set_page_config("Chat with PDF documents", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:

        st.session_state.conversation=None

    if "chat_history" not in st.session_state:

        st.session_state.chat_history=None

    

    
    st.header("Chat with PDF 💬")

    st.subheader("Your Documents")

    pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)


    
    

    if st.button('Process'):

            with st.spinner("Processing"):

            #Extract Text from PDF

                raw_text = uc31_back.get_pdf_text(pdf_docs)

            #Split the Text into Chunks

                text_chunks = uc31_back.get_text_chunks(raw_text)

            #Create Vector Store
                uc31_back.add_to_vector_store(persist_directory,text_chunks,embeddings)

            st.success("Done!")

    user_question = st.text_input("Ask a question from your documents")

    

    if user_question:
        

    
            # Create Conversation Chain
        
        st.session_state.conversation=uc31_back.get_conversation_chain(persist_directory, llm, embeddings)
        handle_user_input(user_question)
