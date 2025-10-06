# ------------------- Import Required Libraries -------------------
# Streamlit for web UI, dotenv for environment variable loading
# ChatOpenAI for LLM interaction, LangChain tools for ReAct agent functionality
# TavilySearchResults for live search, requests for API calls, os for environment variable access
import streamlit as st
from dotenv import load_dotenv
import requests
from httpx import ReadTimeout
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os


# ------------------- Load Environment Variables -------------------
# Automatically loads secrets (API keys) from the .env file into the environment.
# Required for OpenAI, Tavily, and WeatherStack APIs.
load_dotenv()

# Load OpenAI API key (for ChatOpenAI)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load Tavily API key (for web search)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# ------------------- Streamlit Page Configuration -------------------
# Sets the page title, favicon, and layout for the chatbot UI.
st.set_page_config(
    page_title="ReAct Agent Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ------------------- App Title & Description -------------------
# Main heading and a short description shown at the top of the app.
st.title("🤖 ReAct Agent Chatbot")
st.markdown("Ask me anything! I can search the web 🌍 and fetch live weather ⛅")


# ------------------- Define Tools -------------------
# Define a custom tool for fetching current weather using the WeatherStack API.
@tool
def get_current_weather(city: str) -> str:
    """Fetch current weather data for a given city"""
    url = f"https://api.weatherstack.com/current?access_key={os.getenv('WEATHERSTACK_API_KEY')}&query={city}"
    try:
        response = requests.get(url, timeout=5)    # limit timeout to avoid hanging
        data = response.json()
        if "error" in data:
            return f"Weather API error: {data['error'].get('info', 'Unknown error')}"
        return f"The weather in {city} is {data['current']['temperature']}°C, {data['current']['weather_descriptions'][0]}"
    except requests.exceptions.Timeout:
        return "Weather API timeout, please try again."
    except Exception as e:
        return f"Weather API failed: {str(e)}"


# ------------------- Build & Cache ReAct Agent -------------------
# Initializes and caches the agent (LLM + tools + reasoning prompt) for reuse.
# Uses LangChain's ReAct framework to let the LLM reason and call tools step by step.
@st.cache_resource(show_spinner=False)
def build_agent():
    """Initialize the LLM, tools, and ReAct agent executor"""
    
    # Chat LLM with OpenAI (GPT-4o-mini for faster responses)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,   # deterministic responses
        timeout=60,      # allow up to 60 seconds before timing out
        max_retries=3    # retry transient failures
    )
    
    # Web search tool (Tavily API)
    search_tool = TavilySearchResults(max_results=3)
    
    # Load a predefined ReAct reasoning prompt from the LangChain Hub
    prompt = hub.pull("hwchase17/react")
    
    # Build the agent that combines reasoning with tools
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_current_weather],
        prompt=prompt
    )
    
    # Executor to manage agent runs, with safe limits
    executor = AgentExecutor(
        agent=agent,
        tools=[search_tool, get_current_weather],
        verbose=True,              # log reasoning steps in console
        max_iterations=5,          # prevent infinite loops
        max_execution_time=60,     # safety cutoff for long queries
       handle_parsing_errors=lambda e: "I had trouble understanding my reasoning step, let me try again more carefully.", # gracefully handle malformed outputs
    )
    return executor


# Create the agent executor (cached, so runs only once)
agent_executor = build_agent()


# ------------------- Initialize Session State -------------------
# Store chat history in session_state so it persists across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------- Sidebar: New Chat Button -------------------
# Provides a button in the sidebar to reset the chat session.
with st.sidebar:
    st.markdown("## 🔄 Session Control")
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()


# ------------------- Render Chat History -------------------
# Display all past messages (user + assistant) in chronological order.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------- Collect User Input -------------------
# Input box for the user to type a new question/message.
user_input = st.chat_input("Type your message...")


# ------------------- Handle Chat Turn -------------------
# When user sends a message, process it through the ReAct agent and stream response.
if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate assistant response with streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()   
        response_text = ""         
        
        try:
            # Show a spinner while the agent is "thinking"
            with st.spinner("Thinking..."):
                # Stream response chunks from the ReAct agent executor
                for chunk in agent_executor.stream({"input": user_input}):
                    if "output" in chunk:                     
                        response_text += chunk["output"]       
                        placeholder.markdown(response_text)  

        # Error handle   
        except ReadTimeout:
        # If the model or API takes too long, show a friendly timeout message
            response_text = "⏱️ Sorry, the model took too long to respond. Please try again."
            placeholder.markdown(response_text)

        except Exception as e:
            # Catch any other unexpected errors to prevent app crash
            response_text = f"⚠️ Unexpected error: {str(e)}"
            placeholder.markdown(response_text)
        
        # Fallback if no output was generated
        if not response_text:
            response_text = "Sorry, I couldn’t generate a response."
            placeholder.markdown(response_text)
    
    # Save assistant reply in session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})