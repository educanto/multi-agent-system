from dotenv import load_dotenv

from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate, PromptTemplate)
from langchain import hub


load_dotenv()


def format_agent_return(agent_name, agent_outcome):
    if agent_outcome and hasattr(agent_outcome, 'content'):
        output = agent_outcome.content
    elif agent_outcome:
        output = agent_outcome
    else:
        return ""
    return (f"Though: I already called the {agent_name} to delegate a "
            f"task. "
            f"\nObservation: The agent returned: "
            f"\n{output} "
            )


agent_descrip = {
    "institutional_agent": "Responsible for addressing inquiries related to "
                           "company policies and institutional procedures. "
                           "This includes questions about deadlines, "
                           "benefits, and other institutional guidelines. ",
    "recruitment_agent": "Handles all tasks related to candidate selection "
                         "and evaluation, including screening applications, "
                         "assessing qualifications, and providing "
                         "recommendations for hiring decisions. The agent has "
                         "a database with candidates curriculums, so you "
                         "don't need to request to the human candidates "
                         "information. ",
    "calculation_agent": "Performs calculations related to time tracking and "
                         "working hours, including computing total hours "
                         "worked, managing time logs, and performing other "
                         "time-related calculations. "
}

system_instruction = (
    "\nYou are an HR supervisor that manages a conversation between the "
    "following agents: \n"
    + '\n'.join([f"{name}: {desc}" for name, desc in agent_descrip.items()]) +

    "\n\nEach agent will perform a task and respond with their results. Use "
    "the agents result to give a response to the human."

    "\n\nThe answers must be as detailed as possible. Use Markdown for better "
    "visualization."
    
    '\n\nAlways use a json blob by providing an action key and an '
    'action_input key with content between double quotes. '
    
    '\n\nValid "action" values: {members} or "Final Answer"'
    
    "\n\nProvide only ONE action per $JSON_BLOB, as shown: "
    
    "\n\n``` "
    "\n{{ "
    '\n"action": An "agent_name" to call or "Final Answer" to talk to human '
    '\n"action_input": Task description for the agent or response to the human'
    "\n}} "
    "\n``` "
    
    "\n\nThe agent has no memory, so you must contextualize as much as "
    "possible the task and provide all the details and parameters necessary "
    "to solve it standalone. "
    
    "\n\nIf the human didn't provide the necessary parameters for the task, "
    "ask the human with 'Final Answer' for them. If the agent encountered "
    "problems executing the task, ask the human to adjust their request based "
    "on the agent's guidance. "
    
    "\n\nIf the human's request is not related to your agents, gently point "
    "out that your agents do not yet have this capability. Use the json blob! "
    
    "\n\nIf you able able to respond the human directly based on the chat "
    "history, go ahead carefully. Use the json blob! "
    
    "\n\nBegin! Reminder to ALWAYS respond with a valid json blob of a single "
    "action. "
)

prompt_additional_msg = ("Reminder to ALWAYS respond with a valid json blob "
                         "no matter what.")

sys_prompt = SystemMessagePromptTemplate.from_template(system_instruction)

memory = MessagesPlaceholder(variable_name='chat_history')

human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=['agent_scratchpad', 'input'],
    template='{input}\n\n{agent_scratchpad} \n\n' + prompt_additional_msg
)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        sys_prompt,
        memory,
        human_prompt
    ]
).partial(members=", ".join(agent_descrip.keys()))

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

summarization_instructions = (
    "Write a summary of the following "
    "curriculum vitae: "
    "\n{context} "
    "\n\nStart with **Summary of 'person_name' "
    "\nSUMMARY: "
)
summarization_prompt = PromptTemplate.from_template(summarization_instructions)
