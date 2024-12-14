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
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
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