import streamlit as st
import requests
import time
import json
import os

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Keep API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8005/chat")

# Custom CSS
st.markdown("""
<style>
.chat-container {max-width: 800px; margin: auto;}
.user-msg {background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;}
.bot-msg {background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chatbot")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    # Download chat
    if "messages" in st.session_state and st.session_state.messages:
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Download Chat",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json"
        )

# Init history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Typing animation
def type_writer(text, speed=0.02):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(f'<div class="bot-msg">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(speed)

# Display messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking... 🤔"):
        try:
            response = requests.post(API_URL, params={"query": user_input})
            if response.status_code == 200:
                bot_reply = response.json().get("response", "No response")
            else:
                bot_reply = f"Error: {response.status_code}"
        except Exception as e:
            bot_reply = f"Request failed: {e}"

    # Typing effect
    type_writer(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.rerun()