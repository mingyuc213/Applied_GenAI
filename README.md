# A2A Multi-Agent Customer Support System

A sophisticated multi-agent system built with Google's Agent Development Kit (ADK) and the A2A (Agent-to-Agent) protocol, demonstrating intelligent routing, agent coordination, and MCP (Model Context Protocol) integration.

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│       Router Agent (Port 10022)     │
│    • Analyzes query intent          │
│    • Routes to specialist agents    │
│    • Coordinates responses          │
└──────┬──────────────────────┬───────┘
       │                      │
       ▼                      ▼
┌─────────────────┐   ┌──────────────────┐
│  Data Agent     │   │  Support Agent   │
│  (Port 10020)   │   │  (Port 10021)    │
│  • Get customer │   │  • Handle support│
│  • List records │   │  • Create tickets│
│  • Update data  │   │  • Escalate      │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │   MCP Server         │
         │   (Port 8000)        │
         │   • Database access  │
         │   • Tool execution   │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   SQLite Database    │
         │   (support.db)       │
         └─────────────────────┘
```

## 🌟 Features

- **Intelligent Routing**: Router agent analyzes queries and delegates to appropriate specialists
- **Multi-Agent Coordination**: Agents collaborate seamlessly via A2A protocol
- **MCP Integration**: Database operations handled through Model Context Protocol
- **Escalation Support**: Urgent issues routed with appropriate priority
- **Multi-Intent Handling**: Complex queries decomposed and executed in parallel

## 📋 Prerequisites

- Python 3.10+
- Google API Key (for Gemini models)
- Conda or virtual environment

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd HW5_ADK
```

### 2. Create Environment

```bash
conda create -n Applied_GenAI python=3.10
conda activate Applied_GenAI
```

### 3. Install Dependencies

```bash
pip install google-adk
pip install fastmcp
pip install a2a-client
pip install python-dotenv
pip install termcolor
pip install httpx
pip install nest-asyncio
pip install uvicorn
pip install sqlite3
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
MCP_SERVER_URL=http://127.0.0.1:8000/sse
```

## 🎯 Usage

### Start the System (3 Terminal Windows Required)

#### Terminal 1: MCP Server

```bash
conda activate Applied_GenAI
python server.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

#### Terminal 2: Agent Servers

```bash
conda activate Applied_GenAI
python agent_server.py
```

You should see:
```
🚀 Launching A2A Agent System...
✅ Agents running on:
   - Data:    http://127.0.0.1:10020
   - Support: http://127.0.0.1:10021
   - Router:  http://127.0.0.1:10022
```

#### Terminal 3: Run Tests

**Option A: Command-line client**
```bash
conda activate Applied_GenAI
python agent_client.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook
# Open: A2A_Multi_Agent_Demo.ipynb
```

## 📝 Test Scenarios

The system includes 5 comprehensive test scenarios:

### 1. Simple Query
```
Query: "Get customer information for ID 5"
Expected: Single agent retrieves customer data
```

### 2. Coordinated Query
```
Query: "I'm customer 12345 and need help upgrading my account"
Expected: Data agent fetches customer + Support agent provides help
```

### 3. Complex Query
```
Query: "Show me all active customers who have open tickets"
Expected: Multiple MCP calls coordinated across agents
```

### 4. Escalation
```
Query: "I've been charged twice, please refund immediately!"
Expected: Urgent routing to support with high priority
```

### 5. Multi-Intent Query
```
Query: "I am customer 1. Update my email to newemail@example.com and show my ticket history"
Expected: Parallel execution of update + retrieval
```



## 🔧 Key Components

### MCP Server (`server.py`)
- FastMCP-based server with SSE transport
- Tools: `get_customer`, `list_customers`, `update_customer`, `create_ticket`, `get_customer_history`, `get_customers_with_open_tickets`
- Database operations via SQLite

### Agents (`a2aAgents.py`)

**Router Agent**
- Analyzes query intent
- Routes to appropriate specialists
- Coordinates multi-agent responses
- Tools: Remote customer_data and support agents

**Customer Data Agent**
- Database access specialist
- Premium customer recognition
- MCP tool integration
- Skills: Get/list/update customers, ticket history

**Support Agent**
- Customer support specialist
- Ticket creation and management
- Escalation handling
- Skills: Create tickets, handle queries, escalate issues

### Agent Servers (`agent_server.py`)
- Runs all three agents concurrently
- Uses uvicorn with nest_asyncio
- Exposes A2A endpoints on ports 10020-10022

