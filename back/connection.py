import os
import httpx
import json
#from httpx_auth import OAuth2ClientCredentials
from openai import AzureOpenAI as AzureOpenAINative
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

import hashlib
import tiktoken
import unicodedata
from dotenv import load_dotenv

load_dotenv()  # Charge la clé depuis .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#from langchain_mistralai import ChatMistralAI

"""
def update_tiktoken():

    os.environ["TIKTOKEN_CACHE_DIR"] = "/mnt/code/tiktoken_cache"
    blobpath = os.environ['TOKEN_BLOB_PATH']
    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    # validate
    assert os.path.exists(os.path.join(os.environ["TIKTOKEN_CACHE_DIR"], cache_key))


def get_auth():

    update_tiktoken()

    OIDC_ENDPOINT = os.environ["OIDC_ENDPOINT"]
    OIDC_CLIENT_ID = os.environ["OIDC_CLIENT_ID"]
    OIDC_CLIENT_SECRET = os.environ["OIDC_CLIENT_SECRET"]
    OIDC_SCOPE = os.environ["OIDC_SCOPE"]
    oauth2_httpxclient=httpx.Client(verify=False)
    auth=OAuth2ClientCredentials(OIDC_ENDPOINT, client_id=OIDC_CLIENT_ID, client_secret=OIDC_CLIENT_SECRET, scope=OIDC_SCOPE,client=oauth2_httpxclient)

    return auth

"""
def create_openai_native():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def create_llm_chat_langchain(model_name,temperature=0):

    client = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=model_name,
        temperature = temperature,
    )

    return client


def create_llm_langchain(model_name,api_version,temperature=0):

    client = OpenAI(
        openai_api_version=api_version,
        openai_api_key=OPENAI_API_KEY,
        model=model_name,
        temperature = temperature
    )

    return client


def create_embeddings_azureopenai(model):

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=model
    )

    return embeddings

