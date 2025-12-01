# Multi-Agent Customer Support System (LangGraph + MCP + Gemini)

This project implements an intelligent **Multi-Agent System** for customer service automation. It uses **LangGraph** for orchestration, **Google Gemini** for reasoning, and the **Model Context Protocol (MCP)** pattern to securely access a local SQLite database via a RESTful API.

The system demonstrates advanced agentic patterns including **Task Allocation**, **Negotiation/Escalation**, and **Multi-Step Coordination**.

## 🏗️ System Architecture

The system consists of three specialized agents connected via a stateful LangGraph workflow:

1.  **🤖 Router Agent (Orchestrator)**
    * **Role:** The entry point for all queries.
    * **Intelligence:** Uses Gemini to classify user intent (Data Retrieval, Coordination, or General Support).
    * **Action:** Routes the task to the Data Agent or Support Agent.

2.  **🗄️ Customer Data Agent (Specialist)**
    * **Role:** The tool execution layer.
    * **Tools:** Connects to a local **MCP Server** (FastAPI) to perform CRUD operations on `support.db`.
    * **Capabilities:** `get_customer`, `update_customer`, `create_ticket`, `get_customer_history`, `list_customers`.
    * **Tech:** Uses LangChain's `AgentExecutor` to strictly follow tool schemas.

3.  **💬 Support Agent (Specialist)**
    * **Role:** The communication layer.
    * **Action:** Synthesizes raw JSON data from the Data Agent into professional, empathetic natural language responses.

## 📋 Prerequisites

* **Python 3.10+**
* **Google Gemini API Key** (Get one [here](https://aistudio.google.com/app/apikey))
* **Ngrok** (Optional: Only if you want to tunnel the MCP server; otherwise, localhost works)

## 🛠️ Installation & Setup

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
```bash
# Create a .env file in the root directory
GOOGLE_API_KEY="your_actual_api_key_here"
# Optional: NGROK_AUTHTOKEN="your_token"
```

### Database Setup
```bash
python database_setup.py
```

## 🚀 Usage
You need to run two separate processes (terminals) to start the system.

### Terminal 1: Start the MCP Server
This starts the FastAPI server that exposes the database tools.

```bash
python mcp_server.py
```

Note: The server runs on http://0.0.0.0:8000.

Important: Check the output! If you are using ngrok, copy the public URL (e.g., https://xyz.ngrok-free.app). If running locally, you will use http://127.0.0.1:8000.

### Terminal 2: Run the Agent Client
This runs the LangGraph workflow and executes the test scenarios.

1. Update agents.py Configuration: Open agents.py and ensure the MCP_URL matches your server from Terminal 1.
2. Run the Test Suite:
```bash
python test_client.py
```



