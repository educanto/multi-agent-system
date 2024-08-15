from typing import Literal, TypedDict, Union

from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents.stuff import (
    create_stuff_documents_chain)
from langchain.chains import create_retrieval_chain
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import format_agent_return

# Path to the institutional documents
inst_docs_path = 'institutional_docs/XYZ Company Onboarding Manual.pdf'


# Define a typed dictionary to represent the state of an agent
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    current_agent: str


# Function to create the supervisor chain, which manages agent execution
def create_supervisor_chain(model, prompt):
    chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_agent_return(
                    x['current_agent'], x['agent_outcome'])
            )
            | prompt
            | model.bind(stop=["Observation: "])
            | JSONAgentOutputParser()
    )
    return chain


# Function to create a retriever for document search using FAISS and text
# splitting
def create_retriever(embedding_model, doc_path):
    loader = PyMuPDFLoader(doc_path)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    docs = text_splitter.split_documents(raw_docs)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()
    return retriever


# Function to create the workflow for agent management and task delegation
def create_workflow(models, tools, prompts):
    # Function to determine which agent should continue based on the
    # supervisor's decision
    def supervisor_should_continue(state: AgentState) -> (
            Literal)["institutional_agent", "recruitment_agent",
    "calculation_agent", END]:
        if isinstance(state["agent_outcome"], AgentFinish):
            return END
        else:
            return state["agent_outcome"].tool

    # Function to run the recruitment agent
    def run_recruitment_agent(state: AgentState):
        input_text = {"messages": [("user", state["agent_outcome"].log)]}
        output = recruitment_agent.invoke(input_text)
        return {"agent_outcome": output['messages'][-1],
                "current_agent": "recruitment_agent"}

    # Function to run the institutional agent
    def run_institutional_agent(state: AgentState):
        input_text = {"input": state["agent_outcome"].tool_input}
        output = institutional_agent.invoke(input_text)
        return {"agent_outcome": output['answer'],
                "current_agent": "institutional_agent"}

    # Function to run the calculation agent
    def run_calculation_agent(state: AgentState):
        input_text = {"messages": [("user", state["agent_outcome"].log)]}
        output = calculation_agent.invoke(input_text)
        return {"agent_outcome": output['messages'][-1],
                "current_agent": "calculation_agent"}

    # Function to run the supervisor
    def run_supervisor(state: AgentState):
        output = supervisor_chain.invoke(state)
        return {"agent_outcome": output}

    # Create the supervisor chain with the provided model and prompt
    supervisor_chain = create_supervisor_chain(models['supervisor'],
                                               prompts['supervisor'])

    # Create a graph that works with a chat model that utilizes tool calling
    recruitment_agent = create_react_agent(
        models["recruitment"],
        tools["recruitment"]
    )

    # Create the retriever for the institutional agent
    retriever = create_retriever(models["retriever"], inst_docs_path)
    institutional_agent = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(models['institutional'],
                                     prompts['institutional'])
    )

    # Create a graph that works with a chat model that utilizes tool calling
    calculation_agent = create_react_agent(
        models["calculation"],
        tools["calculation"]
    )

    # Create the state graph for managing the workflow
    workflow = StateGraph(AgentState)

    # Add nodes for each agent and the supervisor
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("institutional_agent", run_institutional_agent)
    workflow.add_node("recruitment_agent", run_recruitment_agent)
    workflow.add_node("calculation_agent", run_calculation_agent)

    # Set the entry point of the workflow to the supervisor
    workflow.set_entry_point("supervisor")

    # Add conditional edges for transitioning between agents
    workflow.add_conditional_edges("supervisor",
                                   supervisor_should_continue)
    workflow.add_edge("institutional_agent", "supervisor")
    workflow.add_edge("recruitment_agent", "supervisor")
    workflow.add_edge("calculation_agent", "supervisor")

    # Compile the workflow into an executable application
    app = workflow.compile()

    return app