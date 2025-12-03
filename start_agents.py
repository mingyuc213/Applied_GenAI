# start_agents.py - Start all A2A agent servers using langgraph dev
# When using langgraph dev, agents with messages key automatically expose /a2a/{assistant_id} endpoints
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    """Start agents using langgraph dev (recommended) or fallback to manual servers"""
    print("=" * 70)
    print("Starting A2A Agent Servers")
    print("=" * 70)
    print("\nNote: For proper A2A support, use 'langgraph dev' which automatically")
    print("exposes /a2a/{assistant_id} endpoints for agents with messages key in state.")
    print("\nTo use langgraph dev, create a langgraph.json configuration file.")
    print("=" * 70)
    
    # Check if langgraph.json exists
    if os.path.exists("langgraph.json"):
        print("\nFound langgraph.json - starting with langgraph dev...")
        try:
            subprocess.run([sys.executable, "-m", "langgraph", "dev"], check=True)
        except subprocess.CalledProcessError:
            print("Error running langgraph dev. Falling back to manual server setup.")
            start_manual_servers()
    else:
        print("\nNo langgraph.json found. Using manual server setup.")
        print("For proper A2A support, create langgraph.json and use 'langgraph dev'")
        start_manual_servers()

def start_manual_servers():
    """Fallback: Start agents manually (for development)"""
    import uvicorn
    from a2a_agents import (
        create_customer_data_agent,
        create_support_agent,
        create_router_agent,
    )
    from fastapi import FastAPI
    from langchain_core.messages import HumanMessage
    
    # Create simple FastAPI servers for each agent
    def create_simple_server(agent_id: str, agent_graph, port: int):
        app = FastAPI(title=f"A2A {agent_id} Agent")
        
        @app.get("/a2a/{agent_id}")
        def get_agent_card(agent_id: str):
            return {
                "agent_id": agent_id,
                "name": f"{agent_id.replace('_', ' ').title()} Agent",
                "description": f"A2A-compatible {agent_id} agent",
                "endpoint": f"/a2a/{agent_id}",
                "capabilities": ["invoke"]
            }
        
        @app.post("/a2a/{agent_id}")
        def invoke_agent(agent_id: str, request: dict):
            try:
                # Handle A2A message/send format
                if "method" in request and request["method"] == "message/send":
                    params = request.get("params", {})
                    message = params.get("message", {})
                    parts = message.get("parts", [])
                    text = parts[0].get("text", "") if parts else ""
                    
                    if not text:
                        raise ValueError("No text content found in message parts")
                    
                    state = {"messages": [HumanMessage(content=text)]}
                    result = agent_graph.invoke(state)
                    response_text = result['messages'][-1].content if result.get('messages') else "No response"
                    
                    return {
                        "jsonrpc": "2.0",
                        "result": {
                            "artifacts": [{
                                "parts": [{"kind": "text", "text": response_text}]
                            }]
                        },
                        "id": request.get("id", "")
                    }
                # Handle A2A invoke format (from test client)
                elif "method" in request and request["method"] == "invoke":
                    params = request.get("params", {})
                    messages = params.get("messages", [])
                    
                    # Extract text from messages
                    text = ""
                    if messages:
                        if isinstance(messages[0], dict):
                            text = messages[0].get("content", "")
                        else:
                            text = str(messages[0])
                    
                    if not text:
                        raise ValueError("No content found in messages")
                    
                    # Ensure we create a proper HumanMessage with content
                    human_msg = HumanMessage(content=text)
                    if not human_msg.content or not human_msg.content.strip():
                        raise ValueError(f"Empty message content: {repr(text)}")
                    
                    state = {"messages": [human_msg]}
                    
                    # Debug: print state before invocation
                    print(f"[DEBUG] Invoking agent {agent_id} with state: messages={len(state['messages'])}, content={state['messages'][0].content[:50]}...")
                    
                    result = agent_graph.invoke(state)
                    
                    # Ensure result has messages
                    if not result.get('messages'):
                        raise ValueError("Agent returned no messages")
                    
                    response_text = result['messages'][-1].content if result.get('messages') else "No response"
                    
                    return {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": response_text,
                            "messages": [{"role": "assistant", "content": response_text}]
                        },
                        "id": request.get("id", 1)
                    }
                else:
                    # Fallback format
                    query = request.get("query", request.get("message", ""))
                    if not query:
                        raise ValueError("No query or message provided")
                    
                    state = {"messages": [HumanMessage(content=query)]}
                    result = agent_graph.invoke(state)
                    response_text = result['messages'][-1].content if result.get('messages') else "No response"
                    return {"response": response_text}
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": str(e)},
                    "id": request.get("id", "")
                }
        
        return app, port
    
    # Create agents
    customer_data_agent = create_customer_data_agent()
    support_agent = create_support_agent()
    router_agent = create_router_agent()
    
    print("\nStarting agents on ports 9300, 9301, 9400...")
    print("Customer Data Agent: http://127.0.0.1:9300/a2a/customer_data")
    print("Support Agent: http://127.0.0.1:9301/a2a/support")
    print("Router Agent: http://127.0.0.1:9400/a2a/router")
    print("\nPress Ctrl+C to stop all servers")
    
    # Note: This is a simplified version. For production, use langgraph dev
    import threading
    
    def run_server(app, port):
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    
    app1, port1 = create_simple_server("customer_data", customer_data_agent, 9300)
    app2, port2 = create_simple_server("support", support_agent, 9301)
    app3, port3 = create_simple_server("router", router_agent, 9400)
    
    t1 = threading.Thread(target=run_server, args=(app1, port1), daemon=True)
    t2 = threading.Thread(target=run_server, args=(app2, port2), daemon=True)
    t3 = threading.Thread(target=run_server, args=(app3, port3), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")

if __name__ == "__main__":
    main()
