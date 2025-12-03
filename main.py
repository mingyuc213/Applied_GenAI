# main.py - A2A Client for Router Agent
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any
import uvicorn

load_dotenv()

# --- Configuration ---
ROUTER_AGENT_URL = os.getenv("ROUTER_AGENT_URL", "http://127.0.0.1:9400")

# --- FastAPI Setup ---
api_app = FastAPI(
    title="A2A Client - Customer Support System",
    description="Client interface for A2A multi-agent customer support system",
    version="1.0.0"
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str

class Response(BaseModel):
    response: str
    messages: List[Dict[str, Any]]

# --- A2A Client Functions ---
def call_router_agent(query: str) -> Dict[str, Any]:
    """Call Router Agent via A2A JSON-RPC protocol"""
    try:
        # JSON-RPC 2.0 format
        payload = {
            "jsonrpc": "2.0",
            "method": "invoke",
            "params": {
                "messages": [{"role": "user", "content": query}]
            },
            "id": 1
        }
        
        response = requests.post(
            f"{ROUTER_AGENT_URL}/a2a/router",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result.get("result", {})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Router Agent: {e}")

# --- API Endpoints ---
@api_app.get("/")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "router_agent_url": ROUTER_AGENT_URL,
        "description": "A2A Client for Multi-Agent Customer Support System"
    }

@api_app.get("/agents")
def list_agents():
    """List available agents"""
    agents = {
        "router": f"{ROUTER_AGENT_URL}/a2a/router",
        "customer_data": "http://127.0.0.1:9300/a2a/customer_data",
        "support": "http://127.0.0.1:9301/a2a/support"
    }
    return {"agents": agents}

@api_app.post("/invoke", response_model=Response)
async def invoke_router(request: QueryRequest):
    """
    Invoke the Router Agent via A2A protocol.
    The Router Agent will coordinate with other agents as needed.
    """
    try:
        result = call_router_agent(request.query)
        
        response_content = result.get("content", "")
        messages = result.get("messages", [])
        
        return Response(
            response=response_content,
            messages=messages
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@api_app.post("/query")
async def query_simple(request: QueryRequest):
    """Simple query endpoint (non-JSON-RPC format)"""
    try:
        # Direct invocation format
        payload = {"query": request.query}
        response = requests.post(
            f"{ROUTER_AGENT_URL}/a2a/router",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Entry Point ---
if __name__ == "__main__":
    CLIENT_PORT = int(os.getenv("CLIENT_PORT", "8500"))
    print("=" * 70)
    print("A2A Client Server Starting")
    print("=" * 70)
    print(f"Router Agent URL: {ROUTER_AGENT_URL}")
    print(f"API Documentation: http://127.0.0.1:{CLIENT_PORT}/docs")
    print(f"Invoke Endpoint: http://127.0.0.1:{CLIENT_PORT}/invoke")
    print("=" * 70)
    uvicorn.run(api_app, host="127.0.0.1", port=CLIENT_PORT)
