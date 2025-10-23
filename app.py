import subprocess
import sys


import time
import streamlit as st
from streamlit_option_menu import option_menu
from front import home, uc11, uc12, uc13, uc21, uc22, uc23, uc31, uc32, uc33, uc34, uc41_front_test
from front import *

import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Chatbot GEN AI")

# On ouvre un run “racine” pour établir le ScriptRunContext
mlflow.start_run(run_name="StreamlitSession", nested=True)

# Maintenant on peut autolog et importer le reste
mlflow.langchain.autolog()


st.set_page_config(layout="wide")
st.title('This is An app test of GEN AI projects with streamlit')
st.write('Use case : ')
st.markdown('Elements of **GEN AI** Use Case')
st.markdown('---')

# Menu de navigation
selected = option_menu(
    menu_title="GEN-AI Navigation",
    options=["Home", "UC1 - Chatbot", "UC2 - Document Analyser"],
    icons=["house", "chat", "database", "robot"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",  # ou "vertical",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "black", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#e9ecef",
        },
        "nav-link-selected": {"background-color": "#007bff"},
    },
)

# Affichage des pages en fonction de l'option choisie
if selected == "Home":
    home.app()
elif selected == "UC1 - Chatbot":
    uc41_front_test.app()  # Tu dois avoir une fonction app() dans uc11.py
elif selected == "UC2 - Document Analyser":
    with st.container():
        st.subheader("UC2 - Document Analyser")
        st.write("This section will contain the document analyser functionalities.")
        # Tu peux ajouter ici les fonctions pour les sous-cas d'utilisation UC2-1, UC2-2, etc.
        uc31.app()
        #uc22.app()
        #uc23.app()
