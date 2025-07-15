import streamlit as st
import logging
from src.agent.executor import create_agent_executor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Smart Supply Chain Assistant", page_icon="🚚", layout="wide")

st.title("🚚 Smart Supply Chain Analysis Assistant")

# Initialize the agent executor in the session state to prevent reloading
if 'agent_executor' not in st.session_state:
    with st.spinner("Initializing AI Agent... Please wait."):
        st.session_state.agent_executor = create_agent_executor()
        st.session_state.messages = []
    st.success("Agent is ready!")

# Display message history
for message in st.session_state.get('messages', []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask a question about the supply chain..."):
    # Add and display the user's message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display the agent's response
    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking... (check the console to see its reasoning)"):
            try:
                response = st.session_state.agent_executor.invoke({"input": query})
                answer = response.get('output', "I could not find an answer.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"An error occurred while running the agent: {e}"
                st.error(error_message)
                logging.error(f"Error during agent execution: {e}", exc_info=True)
                st.session_state.messages.append({"role": "assistant", "content": error_message}) 