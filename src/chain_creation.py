examples = [
    {
        "input": "Get the highest payment value made by any customer.",
        "query": "SELECT MAX(payment_value) FROM order_payments;"
    },
    {
        "input": "Get the count of orders with 'delivered' status.",
        "query": "SELECT COUNT(*) FROM orders WHERE order_status = 'delivered';"
    },
    {
        "input": "Find the average weight of all products.",
        "query": "SELECT AVG(product_weight_g) FROM products;"
    },
    {
        "input": "Calculate the total freight value for each order.",
        "query": "SELECT order_id, SUM(freight_value) AS total_freight FROM order_items GROUP BY order_id;"
    },
    {
        "input": "Count the number of customers in each state.",
        "query": "SELECT customer_state, COUNT(*) FROM customers GROUP BY customer_state;"
    },
    {
        "input": "Get the average review score for all orders.",
        "query": "SELECT AVG(review_score) FROM order_reviews;"
    },
    {
        "input": "Get the number of sellers in each state.",
        "query": "SELECT seller_state, COUNT(*) FROM sellers GROUP BY seller_state;"
    },
    {
        "input": "Count the number of unique zip code prefixes across all locations.",
        "query": "SELECT COUNT(DISTINCT geolocation_zip_code_prefix) FROM geolocation;"
    },
    {
        "input": "Count the number of qualified leads from each landing page.",
        "query": "SELECT landing_page_id, COUNT(*) FROM leads_qualified GROUP BY landing_page_id;"
    },
    {
        "input": "Count the number of closed leads by business segment.",
        "query": "SELECT business_segment, COUNT(*) FROM leads_closed GROUP BY business_segment;"
    }
]

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from src.table_choice import select_table
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.cleaning import clean_sql_query
from dotenv import load_dotenv
import os
# import faiss
from langchain_community.vectorstores import FAISS

load_dotenv()
import streamlit as st

HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def get_example_selector():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        hf,
        FAISS,
        k=2,
        input_keys=["input"],
    )
    return example_selector


def create_chain(llm,db):
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question and if it's in table format structure it properly.

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
        example_selector=get_example_selector(),
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