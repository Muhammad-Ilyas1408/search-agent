# ------------------- Import Required Libraries -------------------
# ChatOpenAI for LLM interaction, Streamlit for web UI,
# LangChain tools for ReAct agent functionality, requests for API calls,
# DuckDuckGoSearchRun for live search, dotenv for environment variable loading
import streamlit as st
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os

# ------------------- Load Environment Variables -------------------
# Automatically loads OpenAI API key and other secrets from .env file
load_dotenv()

# ------------------- Streamlit Page Configuration -------------------
# Sets page title, icon, and layout for the chatbot interface
st.set_page_config(
    page_title="ReAct Agent Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ------------------- App Title & Description -------------------
# Display the main heading and instructions for users
st.title("🤖 ReAct Agent Chatbot")
st.markdown("Ask me anything! I can search the web 🌍 and fetch live weather ⛅")

# ------------------- Define Tools -------------------
# Define a tool to fetch current weather information from a public API
@tool
def get_current_weather(city: str) -> str:
    """Fetch current weather data for a given city"""
    url = f"https://api.weatherstack.com/current?access_key={os.getenv('WEATHERSTACK_API_KEY')}&query={city}"
    try:
        response = requests.get(url, timeout=10)  # Avoid long API hangs
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Weather API timeout"}
    except Exception as e:
        return {"error": str(e)}

# ------------------- Build & Cache ReAct Agent -------------------
# Initialize and cache the ReAct agent for efficient reuse
@st.cache_resource(show_spinner=False)
def build_agent():
    """Initialize the LLM, tools, and ReAct agent executor"""
    llm = ChatOpenAI(timeout=25, max_retries=2)       # Chat model with retries
    search_tool = DuckDuckGoSearchRun(max_results=3) # Web search tool
    prompt = hub.pull("hwchase17/react")              # Predefined ReAct prompt
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_current_weather],
        prompt=prompt
    )
    executor = AgentExecutor(
        agent=agent,
        tools=[search_tool, get_current_weather],
        verbose=True,           # Show execution details in console
        max_iterations=3,       # Prevent infinite reasoning loops
        handle_parsing_errors=True  # Safely handle malformed outputs
    )
    return executor

# Initialize the agent executor
agent_executor = build_agent()

# ------------------- Initialize Session State -------------------
# Store chat messages persistently across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- Sidebar: New Chat Button -------------------
# Allows user to reset chat history and start a fresh conversation
with st.sidebar:
    st.markdown("## 🔄 Session Control")
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------- Render Chat History -------------------
# Display all previous messages in the correct order
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------- Collect User Input -------------------
# Input box for the user to type a new message
user_input = st.chat_input("Type your message...")

# ------------------- Handle Chat Turn -------------------
# Process user input, generate assistant reply, and update chat history
if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate the assistant response while showing a spinner
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        reply = response.get("output", "Sorry, I couldn’t generate a response.")

    # Display assistant reply only after generation is complete
    with st.chat_message("assistant"):
        st.write(reply)

    # Store assistant reply in session state
    st.session_state.messages.append({"role": "assistant", "content": reply})