import streamlit as st

def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"])>0:
        for chat_massage in st.session_state["messages"]:
            st.chat_message(chat_massage.role).write(chat_massage.content) 