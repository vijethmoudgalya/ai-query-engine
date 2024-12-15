import streamlit as st
from src.langchain_utils import invoke_chain
from src.langchain_utils import sql_engine
from src.example_rag import get_example_selector
# from langchain_utils import invoke_chain
import os


# Set OpenAI API key from Streamlit secrets
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# GROQ_API_KEY = st.secrets(['GROQ_API_KEY'])


st.set_page_config(
    page_title="AI query engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def main():
    choice =  st.sidebar.radio("Select your preferred Engine:", ["AI Engine", "SQL Engine"])
    #     st.title("Choose your options...")
        
    # ai_engine_btn =  st.button("AI Engine")
    # sql_engine_btn = st.button("SQL Engine")
    if choice == "AI Engine":
        try:
            st.markdown("<h1 style='text-align: center;'>AI Retail Query Engine</h1>", unsafe_allow_html=True)
            with st.spinner("Loading"):
                example_selector = get_example_selector()
            # Initialize chat history
            
            if "messages" not in st.session_state:
                # print("Creating session state")
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What is up?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.spinner("Generating response..."):
                    with st.chat_message("assistant"):
                        response = invoke_chain(prompt,example_selector,st.session_state.messages)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.exception(e)
            
    if choice == "SQL Engine":
        st.markdown("<h1 style='text-align: center;'>SQL Retail Query Engine</h1>", unsafe_allow_html=True)
  
        text_ip = st.text_input("Enter your query")
        sql_response = sql_engine(text_ip)
        st.dataframe(sql_response)

if __name__ == '__main__':
    main()