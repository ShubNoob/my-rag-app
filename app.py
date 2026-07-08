import streamlit as st
from my_rag_backend import generate_script_from_rag

st.title("📚 Our Team's RAG Assistant")

# 1. Initialize chat history in Streamlit's "session state"
# This keeps the chat from erasing every time you press enter
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Create the input box at the bottom
if prompt := st.chat_input("Lets write a script for OOLKA..."):
    
    # Show what the user typed
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Get the response from your RAG model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            response = generate_script_from_rag(prompt)
            
            st.markdown(response)
    
    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})