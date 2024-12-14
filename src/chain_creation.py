from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.cleaning import clean_sql_query
from dotenv import load_dotenv
import os
# import faiss


load_dotenv()
import streamlit as st

def create_chain(llm,example_selector,db):
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question and if it's in table format structure it properly.
        If there is an error with the query or the result Do Not answer the user question, Just output a message "Error cannot query the database".
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    rephrase_answer = answer_prompt | llm | StrOutputParser()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input","top_k"],
    )
    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries. Those examples are just for referecne and hsould be considered while answering follow up questions"),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
    )
    generate_query = create_sql_query_chain(llm, db,final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    # final_chain = (
    # RunnablePassthrough.assign(table_names_to_use=select_table) |
    # RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)).assign(
    # result=itemgetter("query") | execute_query
    # )
    # | rephrase_answer
    # )
    final_chain = (
    RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)).assign(
    result=itemgetter("query") | execute_query
    )
    | rephrase_answer
    )
    return final_chain