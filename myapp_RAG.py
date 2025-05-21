import os

# Set your API key and proxy settings
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNpsPnICYXO1iGQ9twPNbemmbV8At8qMo"

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyRAGApp/1.0)"

from langchain.chat_models import init_chat_model
response_model = init_chat_model("gemini-2.0-flash", model_provider='google_genai')

import trafilatura
import requests
from langchain.schema import Document

# Prompt user for URL and extract main content
url = input("Enter the URL you want to learn more about: ").strip()
html = requests.get(url).text
main_text = trafilatura.extract(html)
if not main_text:
    exit()
all_docs = [Document(page_content=main_text)]

if not all_docs:
    exit()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
docs_split = text_splitter.split_documents(all_docs)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

from langchain_core.vectorstores import InMemoryVectorStore
memory_store = InMemoryVectorStore.from_documents(documents=docs_split, embedding=embeddings)
retriever = memory_store.as_retriever()

def retrieve_and_append(state):
    query = state["messages"][0].content
    results = retriever.invoke(query)
    if isinstance(results, list):
        context = "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        context = results.page_content if hasattr(results, "page_content") else str(results)
    new_messages = state["messages"] + [{"role": "system", "content": context}]
    return {"messages": new_messages}

from langgraph.graph import MessagesState

def generate_query_or_respond(state: MessagesState):
    return {"messages": state["messages"]}

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, return 'yes'. Otherwise return 'no'."
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' is not relevant"
    )

grader_model = init_chat_model("gemini-2.0-flash", model_provider='google_genai')

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (grader_model.with_structured_output(GradeDocuments)
                            .invoke([{"role": "user", "content": prompt}]))

    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n -------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", retrieve_and_append)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_edge("generate_query_or_respond", "retrieve")
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()

while True:
    user_question = input("\nEnter your question about the document (or type 'exit' to quit): ").strip()
    if user_question.lower() == "exit":
        break
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]
        }
    ):
        for node, update in chunk.items():
            last_msg = update["messages"][-1]
            if isinstance(last_msg, dict):
                print(last_msg.get('content', ''))
            else:
                print(getattr(last_msg, 'content', str(last_msg)))
