import asyncio
import threading
import time
import uvicorn
import nest_asyncio
from termcolor import colored

# Import ADK server components
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

# Import GLOBAL objects from a2aAgents.py
from a2aAgents import (
    data_agent, data_agent_card,
    support_agent, support_agent_card,
    router_agent, router_agent_card
)

# Crucial for running multiple servers in one script
nest_asyncio.apply()

def create_agent_a2a_server(agent, agent_card):
    """Factory to create the Starlette app for an agent."""
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    config = A2aAgentExecutorConfig()
    executor = A2aAgentExecutor(runner=runner, config=config)
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

async def run_agent_server(agent, agent_card, port) -> None:
    """Runs a single uvicorn server for an agent."""
    app = create_agent_a2a_server(agent, agent_card)
    
    config = uvicorn.Config(
        app.build(),
        host='127.0.0.1',
        port=port,
        log_level='warning',
        loop='none', # Use the existing loop provided by nest_asyncio
    )
    server = uvicorn.Server(config)
    await server.serve()

async def start_all_servers() -> None:
    print(colored("🚀 Launching A2A Agent System...", "magenta"))
    
    tasks = [
        asyncio.create_task(run_agent_server(data_agent, data_agent_card, 10020)),
        asyncio.create_task(run_agent_server(support_agent, support_agent_card, 10021)),
        asyncio.create_task(run_agent_server(router_agent, router_agent_card, 10022)),
    ]
    
    # Wait briefly for servers to spin up
    await asyncio.sleep(2)
    
    print(colored("✅ Agents running on:", "green"))
    print(colored("   - Data:    http://127.0.0.1:10020", "cyan"))
    print(colored("   - Support: http://127.0.0.1:10021", "cyan"))
    print(colored("   - Router:  http://127.0.0.1:10022", "cyan"))
    print(colored("⚠️  Press Ctrl+C to stop.", "yellow"))
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Servers cancelled.")
    except Exception as e:
        print(f"Server error: {e}")

def run_servers_in_background() -> None:
    """Runs the main asyncio loop in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_all_servers())

if __name__ == "__main__":
    # Start the server thread
    t = threading.Thread(target=run_servers_in_background, daemon=True)
    t.start()
    
    # Keep the main thread alive so the daemon thread doesn't die
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(colored("\n🛑 Stopping...", "red"))