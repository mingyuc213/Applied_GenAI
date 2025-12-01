# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import List
import uvicorn

# Load the compiled graph from agents.py
from agents import app, AgentState

# Load environment variables (API Key)
load_dotenv()

# --- 1. FastAPI Setup ---
api_app = FastAPI(
    title="LangGraph A2A Router",
    description="Customer Support Multi-Agent Orchestrator",
    version="1.0"
)

# --- 2. Request Schema ---
class QueryRequest(BaseModel):
    query: str

class Response(BaseModel):
    response: str
    messages: List[str]


# --- 3. Entry Point for Client Communication ---
@api_app.post("/invoke", response_model=Response)
async def invoke_graph(request: QueryRequest):
    """
    Invokes the compiled LangGraph with a user query.
    Simulates the A2A client interaction with the Router Agent.
    """
    
    # Initial state containing the user message
    initial_state = {"messages": [HumanMessage(content=request.query)]}
    
    # Invoke the LangGraph application
    # The graph runs synchronously for this example.
    final_state: AgentState = app.invoke(initial_state)

    # Extract the final response message
    final_response = final_state['messages'][-1].content
    
    # Optional: Log the full message history for debugging
    message_history = [m.content for m in final_state['messages']]

    return Response(
        response=final_response,
        messages=message_history
    )

# --- 4. Uvicorn Runner (Execute this file) ---
if __name__ == "__main__":
    # You will access the service at: http://127.0.0.1:8000/invoke
    # and the documentation at: http://127.0.0.1:8000/docs
    print("Starting FastAPI/Uvicorn server...")
    uvicorn.run(api_app, host="127.0.0.1", port=8000)