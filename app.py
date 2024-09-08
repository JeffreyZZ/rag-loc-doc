#!/bin/env python3
import os, time
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF
from chatmarkdown import ChatMarkdown
from elasticsearch import Elasticsearch, ConflictError

st.set_page_config(page_title="ChatMarkdown")

elasticsearch_url = "http://localhost:9200"
index_name = "rag-loc-doc"

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # TODO: need to clean up Elasticsearch and vector store before ingestion starts.
    client = Elasticsearch(elasticsearch_url)
    exists = client.indices.exists(index=index_name)
    if exists:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # delete all documents
                client.delete_by_query(
                    index=index_name,
                        body={
                            "query": {
                                "match_all": {}
                            }
                        },
                        wait_for_completion=True
                    )
                break
            except ConflictError as e:
                print(f"ConflictError encountered: {e.info}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                        
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            is_ingested = st.session_state["assistant"].ingest(file_path, os.path.splitext(file.name)[0])
            if not is_ingested:
                st.session_state["messages"].append((f"{file.name} has no content.", False))
            time.sleep(0.1)    
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatMarkdown()

    st.header("ChatMarkdown")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["md"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
