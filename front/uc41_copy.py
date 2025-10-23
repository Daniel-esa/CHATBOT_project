# ✅ Fichier 2 : front/uc41___.py (Frontend RAG)

import streamlit as st
from back import uc41_back_copy, connection
from css.uc31html import css, bot_template, user_template
from dotenv import load_dotenv, find_dotenv

AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"


def app():
    st.write(css, unsafe_allow_html=True)
    st.header("P&P Chatbot")
    st.subheader("You are reading Spend and Procurement Savings Reports")

    # \uD83D\uDD01 Initialisation de l'historique de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # \uD83D\uDCE6 Chargement des modèles
    llm = connection.create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O, AZURE_AOAI_API_VERSION, 0.7)
    embeddings = connection.create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL, AZURE_AOAI_API_VERSION)
    uc41_back_copy.initialize_chain(llm, embeddings)

    # \uD83D\uDC4B Message d'accueil
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("Bonjour, Je suis votre assistant Spend & Savings. Comment puis-je vous aider ?")

    # \uD83D\uDCAC Entrée utilisateur
    user_question = st.chat_input("Pose ta question sur le spend ou les savings...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Formatage de l'historique pour le prompt
        chat_history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.chat_history
            if msg["role"] in ("user", "assistant")
        )

        # Affichage du contexte utilisé
        debug_context = uc41_back_copy.get_context_with_filters(user_question)
        with st.expander(" Contexte RAG utilisé (debug)"):
            st.code(debug_context["context"][:3000])  # Affiche 3000 premiers caractères max

        # Appel du RAG
        response = uc41_back_copy.rag_chain.invoke({
            "question": user_question,
            "chat_history": chat_history_text
        })

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Affichage du chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
