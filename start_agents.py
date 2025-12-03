# start_agents.py - Start all A2A agent servers
import os
import threading
import time
import uvicorn
from dotenv import load_dotenv
from a2a_agents import (
    create_customer_data_agent,
    create_support_agent,
    create_router_agent,
    create_agent_server,
    AGENT_REGISTRY
)

load_dotenv()

# Agent ports
CUSTOMER_DATA_PORT = 9300
SUPPORT_PORT = 9301
ROUTER_PORT = 9400

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1")

def start_agent_server(agent_id: str, agent_graph, port: int):
    """Start an agent server in a separate thread"""
    app, _ = create_agent_server(agent_id, agent_graph, port)
    
    # Register agent in registry
    AGENT_REGISTRY[agent_id] = f"{BASE_URL}:{port}"
    print(f"Registered {agent_id} agent at {AGENT_REGISTRY[agent_id]}")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

def main():
    """Start all agent servers"""
    print("=" * 70)
    print("Starting A2A Agent Servers")
    print("=" * 70)
    
    # Create agent graphs
    customer_data_agent = create_customer_data_agent()
    support_agent = create_support_agent()
    router_agent = create_router_agent()
    
    # Start agents in separate threads
    threads = []
    
    # Customer Data Agent
    t1 = threading.Thread(
        target=start_agent_server,
        args=("customer_data", customer_data_agent, CUSTOMER_DATA_PORT),
        daemon=True
    )
    t1.start()
    threads.append(t1)
    time.sleep(1)  # Give server time to start
    
    # Support Agent
    t2 = threading.Thread(
        target=start_agent_server,
        args=("support", support_agent, SUPPORT_PORT),
        daemon=True
    )
    t2.start()
    threads.append(t2)
    time.sleep(1)
    
    # Router Agent
    t3 = threading.Thread(
        target=start_agent_server,
        args=("router", router_agent, ROUTER_PORT),
        daemon=True
    )
    t3.start()
    threads.append(t3)
    time.sleep(2)  # Give all servers time to register
    
    print("\n" + "=" * 70)
    print("All A2A Agent Servers Started")
    print("=" * 70)
    print(f"Customer Data Agent: http://127.0.0.1:{CUSTOMER_DATA_PORT}/a2a/customer_data")
    print(f"Support Agent: http://127.0.0.1:{SUPPORT_PORT}/a2a/support")
    print(f"Router Agent: http://127.0.0.1:{ROUTER_PORT}/a2a/router")
    print("\nAgent Registry:")
    for agent_id, url in AGENT_REGISTRY.items():
        print(f"  {agent_id}: {url}")
    print("\nPress Ctrl+C to stop all servers")
    print("=" * 70)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")

if __name__ == "__main__":
    main()

