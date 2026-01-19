# puls-events-chatbot-intelligent-rag/dashboard/app.py

import streamlit as st
import requests

API_URL = "http://rag-api:8000"

st.set_page_config(page_title="RAG Monitor", layout="wide")

st.title("📊 RAG Monitoring Dashboard")

query = st.text_input("Ask something:")

if st.button("Ask RAG"):
    r = requests.post(f"{API_URL}/ask", json={"question": query})
    st.write(r.json()["answer"])

st.divider()

if st.button("Run RAG Evaluation (RAGAS)"):
    r = requests.post(f"{API_URL}/evaluate")
    st.json(r.json())

st.divider()

if st.button("Rebuild Vector Store"):
    r = requests.post(f"{API_URL}/rebuild")
    st.success("Vector store rebuilt")
