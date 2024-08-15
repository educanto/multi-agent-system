import re
from dotenv import load_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import config

from tools import WorkingHoursCalculator, SummarizeCVs, CalculatorTool
from prompts import supervisor_prompt, retrieval_qa_chat_prompt
from agents import create_workflow

# Set the default recursion limit for LangChain
config.DEFAULT_RECURSION_LIMIT = 10

# Load environment variables from a .env file
load_dotenv()

# Define user and bot names
user_name = 'You'
bot_name = 'Guilda Agent'
avatars = {"human": user_name, "ai": bot_name}

# Set up the Streamlit page configuration
st.set_page_config(
    page_title=f"{bot_name}"
)
st.subheader(f"I'm {bot_name}, HR Specialist")

# Display an informational message with details about the application
st.info(
    (
        "Welcome to the HR Specialist! This application is designed to assist "
        "with various HR-related tasks. Here’s how you can benefit from our "
        "specialized agents:"

        "\n\n**Institutional Agent**: Handles inquiries related to company "
        "policies and institutional procedures. Get answers to questions "
        "about deadlines, benefits, and other HR guidelines. "

        "\n\n**Recruitment Agent**: Manages the recruitment process by "
        "handling candidate selection and evaluation. This includes screening "
        "applications, assessing qualifications, and making hiring "
        "recommendations."

        "\n\n**Calculation Agent**: Performs calculations related to time "
        "tracking and working hours. It helps compute total hours worked, "
        "manage time logs, and carry out other time-related calculations."

        "\n\nUse these agents to streamline HR tasks, manage your workforce, "
        "and ensure efficient operations! "
    ),
    icon="ℹ️",
)


# Function to escape dollar signs for correct formatting in Streamlit
def prepare_formatting(text):
    return re.sub(r"(?<!\\)\$", r"\\$", text)


# Function to add a message to the chat history with proper formatting
def add_message_to_chat_history(chat_hist, message):
    message.content = prepare_formatting(message.content)
    chat_hist.append(message)


# Initialize chat memory if it doesn't exist
if "memory" not in st.session_state:
    st.session_state.memory = [AIMessage("Hi! How can I assist you today?")]

# Display chat history
for msg in st.session_state.memory:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Initialize the execution graph (workflow) if it doesn't exist
if "execution_graph" not in st.session_state:
    # Define the models and tools for different agents
    model = ChatOpenAI(model="gpt-4o-mini")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    models = {
        "institutional": model,
        "recruitment": model,
        "calculation": model,
        "supervisor": model,
        "retriever": embedding_model
    }

    tools = {
        "recruitment": [SummarizeCVs()],
        "calculation": [WorkingHoursCalculator(), CalculatorTool()]
    }

    prompts = {
        "institutional": retrieval_qa_chat_prompt,
        "supervisor": supervisor_prompt
    }

    # Create the workflow using the models, tools, and prompts
    workflow = create_workflow(models, tools, prompts)

    # Store the workflow in the session state
    st.session_state.execution_graph = workflow

    # Show balloons animation to indicate the setup is complete
    st.balloons()

# Process user input from the chat interface
if user_input := st.chat_input("Tip here"):
    st.chat_message("human").write(user_input)

    # Prepare inputs for the execution graph
    inputs = {"input": user_input, "chat_history": st.session_state.memory}

    # Invoke the workflow and get the response
    response = st.session_state.execution_graph.invoke(inputs)
    output_text = response['agent_outcome'].return_values['output']

    # Debug information for devs
    # st.write(response)

    # Display the bot's response in the chat interface
    st.chat_message(bot_name).write(output_text)

    # Update chat history with the new messages
    st.session_state.memory.append(HumanMessage(user_input))
    st.session_state.memory.append(AIMessage(output_text))
