import os
from datetime import datetime
from typing import List, Any
from dotenv import load_dotenv

from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.combine_documents.stuff import (
    create_stuff_documents_chain)
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader

from prompts import summarization_prompt


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
llm_math = LLMMathChain.from_llm(llm)
cvs_directory = 'cvs_docs'


class WorkingHoursCalculator(BaseTool):
    name = "WorkingHoursCalculator"
    description = (
        "Use this tool to calculate the total working hours based on "
        "user clock punches. The tool can't be used with odd number of "
        "punches. In this case, give it back to human and request full to "
        "check. NEVER invent intervals. MUST Use 24-hour time format. "
    )

    def _run(self, clock_punches: List[str]):
        # Convert clock punch strings to datetime objects
        punches = [datetime.strptime(punch, "%H:%M") for punch in
                   clock_punches]

        # Check if the number of clock punches is odd
        if len(clock_punches) % 2 != 0:
            return ("Could not calculate. The number of clock punches must be "
                    "even. Give it back to human and request user to check.")

        # Check if there are at least two clock punches
        if len(punches) < 2:
            return (
                "Could not calculate. At least two clock punches are required."
                "Give it back to human and request user to check. ")

        total_working_minutes = 0

        # Calculate working time between clock punches
        for i in range(1, len(punches), 2):
            start_time = punches[i - 1]
            end_time = punches[i]

            # Check if start time is before end time
            if start_time >= end_time:
                return ("Could not calculate. Start time must be before end "
                        "time.")

            # Calculate working time between start and end times
            working_period_minutes = (end_time - start_time).seconds // 60
            total_working_minutes += working_period_minutes

        # Convert total working time to hours and minutes
        hours = total_working_minutes // 60
        minutes = total_working_minutes % 60

        return f"{hours} hours and {minutes} minutes"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class SummarizeCVs(BaseTool):
    name = "SummarizeCVs"
    description = ("Use this tool to summarize the curriculum vitae (CV) of "
                   "candidates. It extracts key information from each CV and "
                   "provides a concatenated summary for each candidate, "
                   "helping you identify the most qualified ones. The CVs are "
                   "automatically retrieved from the default directory, so "
                   "there s no need to provide them manually.")

    def _run(self, dummy_arg: str) -> str:
        documents = load_pdfs_from_directory(cvs_directory)
        final_summary = summarize_documents(documents, llm)
        return final_summary

    def _arun(self, dummy_arg: str):
        raise NotImplementedError("This tool does not support async")


class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "Designed to handle mathematical calculations. Use "
    "to provide accurate mathematical answers. "

    def _run(self, query: str) -> str:
        return llm_math.invoke(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")


def load_pdfs_from_directory(path):
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    documents = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(os.path.join(path, pdf_file))
        docs = loader.load()
        documents.append(docs)
    return documents


def summarize_documents(docs_lists, llm):
    stuff_chain = create_stuff_documents_chain(llm, summarization_prompt)
    summaries = []
    for docs in docs_lists:
        summary = stuff_chain.invoke({"context": docs})
        summaries.append(summary)
    return "\n\n---\n\n".join(summaries)
