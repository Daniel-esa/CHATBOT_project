import streamlit as st
from back import uc35_back, connection
from dotenv import load_dotenv, find_dotenv
import os


def app():
    load_dotenv(find_dotenv())
    st.title("AI Assistant for Data Science")
    st.header("Exploratory Data Analysis Part")
    st.write("Hello, I'm your AI Assistant and I'm here to help you with your data science projects")



    with st.sidebar:
        st.write("Your Data Science Adventure Begins with an CSV file") 
        st.caption("""You may alreday know that every exiciting data science journey starts with a dataset.
                That's why I'd love for you to upload a CSV file. Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
                Then, we'll work together to shape your business challenge into a data science framework. I'll introduce you to the coolest machine learning modesl, 
                and we'll use them to tackle your problem. Sounds fun right.
                """)
        st.divider()
        st.caption("<p style='text-align:center'>made with 🏆🎯💎 by Haoshi</p>", unsafe_allow_html=True)


    input_model= st.selectbox(label="Select Model", options=['gpt4o','gpt35turbo'])
    llm = connection.create_llm_chat(input_model,"2024-02-15-preview")

    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1:False}

    def clicked(button):
        st.session_state.clicked[button] = True

    st.button("Let's get started", on_click=clicked, args=[1])

    if st.session_state.clicked[1]:
        tab1, tab2 = st.tabs(["Data Analysis and Data Science","ChatBox"])
        with tab1:
            user_csv = st.file_uploader("Upload file to be summarized, only <b>[ONE]</b> file allowed", type=["xlsx","xlsm","xls", "csv"])
        with tab2:
            st.header("ChatBox")
            st.write("🤖 Welcome to the AI Assistant ChatBox!") 
            st.write("Got burning questions about your data science problem or need help navigating the intricacies of your project? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

