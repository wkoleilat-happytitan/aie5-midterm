from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import chainlit as cl
import operator

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

path = "data/"
loader = DirectoryLoader(path, glob="*.pdf")
docs = loader.load()

tavily_tool = TavilySearchResults(max_results=5)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
split_documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="apis_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="apis_docs",
    embedding=embeddings,
)

_ = vector_store.add_documents(documents=split_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def retrieve(state):
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

RAG_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.
### Question
{question}
### Context
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

llm = ChatOpenAI(model="gpt-4o-mini")

def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = llm.invoke(messages)
  return {"response" : response.content}

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

@tool
def ai_rag_tool(question: str) -> str:
  """Useful for when you need to answer questions about apis. Input should be a fully formed question."""
  response = graph.invoke({"question" : question})
  return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"]
    }

tool_belt = [
    tavily_tool,
    ai_rag_tool
]

model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]
  context: List[Document]

tool_node = ToolNode(tool_belt)

uncompiled_graph = StateGraph(AgentState)

def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {
        "messages": [response],
        "context": state.get("context", [])
    }

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

uncompiled_graph.set_entry_point("agent")

def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

uncompiled_graph.add_edge("action", "agent")

compiled_graph = uncompiled_graph.compile()

@cl.on_chat_start
async def start():
  cl.user_session.set("graph", compiled_graph)

@cl.on_message
async def handle(message: cl.Message):
  graph = cl.user_session.get("graph")
  state = {"messages" : [HumanMessage(content=message.content)]}
  response = await graph.ainvoke(state)
  await cl.Message(content=response["messages"][-1].content).send()