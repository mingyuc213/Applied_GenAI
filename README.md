# Multi-Agent Customer Support System (A2A Protocol + MCP + LangGraph)

This project implements an intelligent **Multi-Agent System** for customer service automation using **Agent-to-Agent (A2A) protocol**, **Model Context Protocol (MCP)**, **LangGraph**, and **Google Gemini**.

The system demonstrates advanced agentic patterns including **Task Allocation**, **Negotiation/Escalation**, and **Multi-Step Coordination** with proper A2A protocol compliance.

## 🏗️ System Architecture

The system consists of three specialized A2A-compatible agents, each running as a separate server:

1.  **🤖 Router Agent (Orchestrator)**
    * **Port:** 9400
    * **A2A Endpoint:** `/a2a/router`
    * **Role:** The entry point for all queries. Analyzes intent and routes to appropriate specialist agents.
    * **Intelligence:** Uses Gemini to classify user intent (Data Retrieval, Coordination, or General Support).
    * **A2A Communication:** Discovers and calls other agents via JSON-RPC protocol.

2.  **🗄️ Customer Data Agent (Specialist)**
    * **Port:** 9300
    * **A2A Endpoint:** `/a2a/customer_data`
    * **Role:** The tool execution layer for database operations.
    * **Tools:** Connects to **MCP Server** to perform CRUD operations on `support.db`.
    * **Capabilities:** `get_customer`, `update_customer`, `create_ticket`, `get_customer_history`, `list_customers`.
    * **MCP Integration:** All database operations go through MCP server tools.

3.  **💬 Support Agent (Specialist)**
    * **Port:** 9301
    * **A2A Endpoint:** `/a2a/support`
    * **Role:** The communication layer that synthesizes responses.
    * **Action:** Provides professional, empathetic natural language responses based on customer queries and data context.

4.  **🔧 MCP Server**
    * **Port:** 8000
    * **Role:** Exposes database tools following MCP protocol specifications.
    * **Endpoints:** `/mcp/tools` (list tools), `/mcp/call_tool` (execute tools)

5.  **🌐 Client API**
    * **Port:** 8500
    * **Role:** Client interface to interact with the multi-agent system.
    * **Endpoints:** `/invoke` (main entry point), `/query` (simple query)

## 📋 Prerequisites

* **Python 3.10+**
* **Google Gemini API Key** (Get one [here](https://aistudio.google.com/app/apikey))
* **Virtual environment** (recommended)

## 🛠️ Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY="your_actual_api_key_here"
MCP_URL="http://127.0.0.1:8000/mcp"
BASE_URL="http://127.0.0.1"
CLIENT_URL="http://127.0.0.1:8500"
ROUTER_AGENT_URL="http://127.0.0.1:9400"
```

### 4. Database Setup
```bash
python database_setup.py
# When prompted, enter 'y' to insert sample data
```

## 🚀 Usage

You need to run **three separate processes** (terminals) to start the complete system.

### Terminal 1: Start the MCP Server
This starts the MCP protocol-compliant server that exposes database tools.

```bash
python mcp_server.py
```

The server runs on `http://127.0.0.1:8000`.

**MCP Endpoints:**
- `GET /mcp/tools` - List all available tools
- `POST /mcp/call_tool` - Execute a tool (JSON-RPC format)

### Terminal 2: Start All A2A Agent Servers
This starts all three A2A-compatible agent servers.

```bash
python start_agents.py
```

This will start:
- Customer Data Agent on `http://127.0.0.1:9300`
- Support Agent on `http://127.0.0.1:9301`
- Router Agent on `http://127.0.0.1:9400`

Each agent exposes:
- `GET /a2a/{agent_id}` - Agent Card (A2A discovery endpoint)
- `POST /a2a/{agent_id}` - Invoke agent (A2A JSON-RPC endpoint)

### Terminal 3: Start Client API (Optional) or Run Tests
You can either start the client API server or run tests directly.

**Option A: Start Client API**
```bash
python main.py
```

The client API runs on `http://127.0.0.1:8500`.

**Option B: Run Test Suite Directly**
```bash
python test_client.py
```

## 🧪 Test Scenarios

The system includes 5 test scenarios that demonstrate different coordination patterns:

1. **Simple Query** - Direct data retrieval: "Get customer information for ID 1"
2. **Coordinated Query** - Task allocation: "I'm customer 2 and need help upgrading my account"
3. **Complex Query** - Multi-step coordination: "Show me all active customers who have open tickets"
4. **Escalation** - Negotiation pattern: "I've been charged twice (Customer ID 1), please refund immediately!"
5. **Multi-Step** - Complex coordination: "Update my email to new@email.com for customer ID 5 and show my ticket history"

## 📡 A2A Protocol Implementation

### Agent Discovery
Each agent exposes an Agent Card at `/a2a/{assistant_id}` that provides:
- Agent ID and name
- Description and capabilities
- Endpoint information

### Agent Communication
Agents communicate via JSON-RPC 2.0 protocol:
```json
{
  "jsonrpc": "2.0",
  "method": "invoke",
  "params": {
    "messages": [{"role": "user", "content": "query"}]
  },
  "id": 1
}
```

### State Management
All agents use LangGraph with message-based state (required `messages` key):
```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
```

## 🔧 MCP Protocol Implementation

### Tool Definitions
All tools follow MCP protocol with JSON Schema definitions:
- `get_customer(customer_id)` - Uses `customers.id`
- `list_customers(status, limit)` - Uses `customers.status`
- `update_customer(customer_id, data)` - Uses `customers` fields
- `create_ticket(customer_id, issue, priority)` - Uses `tickets` fields
- `get_customer_history(customer_id)` - Uses `tickets.customer_id`

### Tool Invocation
Tools are called via `/mcp/call_tool` endpoint:
```json
{
  "name": "get_customer",
  "arguments": {"customer_id": 1}
}
```

## 📁 Project Structure

```
HW5/
├── mcp_server.py           # MCP protocol-compliant server
├── a2a_agents.py           # A2A-compatible agent definitions
├── start_agents.py         # Script to start all agent servers
├── main.py                 # Client API server
├── test_client.py          # Test suite
├── database_setup.py       # Database initialization
├── requirements.txt        # Python dependencies
├── support.db             # SQLite database (auto-created)
└── README.md              # This file
```

## 🔍 Key Features

✅ **A2A Protocol Compliance**
- Each agent is a separate server with `/a2a/{assistant_id}` endpoint
- Agents discover and communicate via JSON-RPC 2.0
- LangGraph message-based state structure

✅ **MCP Protocol Compliance**
- Proper tool definitions with JSON Schema
- `/mcp/tools` and `/mcp/call_tool` endpoints
- All database operations via MCP tools

✅ **Three Coordination Scenarios**
- Task Allocation: Router delegates to appropriate specialist
- Negotiation: Multi-agent coordination for complex requests
- Multi-Step: Sequential agent handoffs with context passing

## 🐛 Troubleshooting

### Agents Not Responding
- Ensure MCP server is running on port 8000
- Check that all agent servers started successfully (ports 9300, 9301, 9400)
- Verify agent registry in `start_agents.py` output

### MCP Server Not Reachable
- Check that database file `support.db` exists
- Run `database_setup.py` if database is missing
- Verify MCP_URL in `.env` matches server URL

### Port Conflicts
- MCP Server: 8000
- Client API: 8500
- Customer Data Agent: 9300
- Support Agent: 9301
- Router Agent: 9400

Change ports in respective files or `.env` if conflicts occur.

## 📝 Notes

- All agents use MCP tools for database operations (no direct database access)
- Customer IDs in test data: 1, 2, 3, 4, 5, and others
- The system demonstrates proper A2A protocol with agent discovery and JSON-RPC communication
- MCP server follows protocol specifications with proper tool definitions

## 📚 References

- [LangGraph A2A Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
