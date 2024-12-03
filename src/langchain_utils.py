import os
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from src.chain_creation import create_chain
from src.cleaning import clean_sql_query
import streamlit as st
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
@st.cache_resource
def get_chain():
    print("Creating chain")
    db = SQLDatabase.from_uri(f"sqlite:///src/data/olist.sqlite")
    llm = ChatGroq(model="llama3-8b-8192")
    
    chain = create_chain(llm,db)
    print("Chain created successfully")
    return chain,db

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question,messages):
    chain,_ = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question,"top_k":3,"messages":history.messages})
    # response = chain.invoke({"question": question,"messages":history.messages})
    
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
def sql_engine(sql_query):
    try:
        conn = sqlite3.connect('./src/data/olist.sqlite')
        cursor = conn.cursor()
        cursor.execute(sql_query)
        # Fetch the results
        data = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        result = pd.DataFrame(data, columns=columns)
     
        return result
    except Exception as e:
        st.error("Error executing sql engine")
    
    


