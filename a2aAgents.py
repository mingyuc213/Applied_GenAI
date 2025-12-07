import os
from dotenv import load_dotenv

# Google ADK Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.mcp_tool import McpToolset, SseConnectionParams

# A2A Protocol Imports
from a2a.types import AgentCard, AgentCapabilities, TransportProtocol, AgentSkill
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

# Configuration
load_dotenv()
MODEL_NAME = "gemini-2.5-flash-lite"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google API Key: ")

# ==============================================================================
# 1. CUSTOMER DATA AGENT
# ==============================================================================
data_agent = Agent(
    model=MODEL_NAME,
    name="customer_data_agent",
    instruction="""
    You are the Customer Data Agent. Your role is to access and manage customer database information via MCP tools.
    
    Your responsibilities:
    - Retrieve customer information by ID
    - List customers with optional status filtering
    - Update customer records
    - Get customer ticket history
    - Get customers with open tickets
    
    Premium / VIP customers: IDs 1 and 12345. Whenever their data is requested,
    explicitly mention that they are premium customers so the Router can route accordingly.
    
    You MUST use your MCP tools to access the database. Do not answer from your own knowledge.
    Always validate data before returning it.
    """,
    tools=[McpToolset(connection_params=SseConnectionParams(url=MCP_SERVER_URL))]
)

data_agent_card = AgentCard(
    name="Customer Data Agent",
    url="http://localhost:10020",
    description="Specialist agent for accessing and managing customer database information via MCP tools",
    version="1.0",
    capabilities=AgentCapabilities(streaming=True),
    preferred_transport=TransportProtocol.jsonrpc,
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id='get_customer_info',
            name='Get Customer Information',
            description='Retrieves customer details by ID using customers.id field',
            tags=['customer', 'data', 'retrieval', 'mcp'],
            examples=['Get customer information for ID 1', 'Retrieve customer 12345']
        ),
        AgentSkill(
            id='list_customers',
            name='List Customers',
            description='Lists customers with optional status filtering using customers.status field',
            tags=['customer', 'list', 'filter', 'mcp'],
            examples=['List all active customers', 'Show me customers with disabled status']
        ),
        AgentSkill(
            id='update_customer',
            name='Update Customer',
            description='Updates customer records using customers fields',
            tags=['customer', 'update', 'modify', 'mcp'],
            examples=['Update email for customer 1']
        ),
        AgentSkill(
            id='get_customer_history',
            name='Get Customer History',
            description='Retrieves ticket history for a customer using tickets.customer_id field',
            tags=['customer', 'history', 'tickets', 'mcp'],
            examples=['Show ticket history for customer 1']
        ),
        AgentSkill(
            id='get_customers_with_open_tickets',
            name='Get Customers with Open Tickets',
            description='Finds customers who have open tickets, optionally filtered by status',
            tags=['customer', 'tickets', 'query', 'mcp'],
            examples=['Show active customers with open tickets']
        ),
    ]
)


# ==============================================================================
# 2. SUPPORT AGENT
# ==============================================================================
support_agent = Agent(
    model=MODEL_NAME,
    name="support_agent",
    instruction="""
    You are the Support Agent. Your role is to handle customer support queries and issues.
    
    Your responsibilities:
    - Handle general customer support queries
    - Create support tickets for customer issues
    - Escalate complex issues when needed
    - Request customer context from Data Agent when needed
    - Provide solutions and recommendations
    
    You have access to customer lookup tools to find customer IDs when needed.
    You can create tickets and check customer history.
    
    If you cannot proceed (e.g., need billing context), tell the Router exactly what information you require.
    For urgent issues (billing, refunds, critical problems), prioritize them appropriately.
    """,
    tools=[McpToolset(connection_params=SseConnectionParams(url=MCP_SERVER_URL))]
)

support_agent_card = AgentCard(
    name="Support Agent",
    url="http://localhost:10021",
    description="Specialist agent for handling customer support queries, ticket creation, and issue resolution",
    version="1.0",
    capabilities=AgentCapabilities(streaming=True),
    preferred_transport=TransportProtocol.jsonrpc,
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id='create_ticket',
            name='Create Support Ticket',
            description='Creates a new support ticket using tickets fields',
            tags=['support', 'ticket', 'create', 'mcp'],
            examples=['Create a ticket for customer 1 about account upgrade']
        ),
        AgentSkill(
            id='handle_support_query',
            name='Handle Support Query',
            description='Processes general customer support queries and provides solutions',
            tags=['support', 'help', 'assistance'],
            examples=['I need help with my account', 'How do I upgrade my subscription?']
        ),
        AgentSkill(
            id='escalate_issue',
            name='Escalate Issue',
            description='Escalates complex or urgent issues appropriately',
            tags=['support', 'escalation', 'urgent'],
            examples=['I\'ve been charged twice, please refund immediately!']
        ),
    ]
)


# ==============================================================================
# 3. ROUTER AGENT (ORCHESTRATOR)
# ==============================================================================

# Define remote agents using RemoteA2aAgent
# This replaces the manual HTTP helper function
remote_customer_data_agent = RemoteA2aAgent(
    name='customer_data',
    description='Specialist agent for accessing customer database information',
    agent_card=f'http://localhost:10020{AGENT_CARD_WELL_KNOWN_PATH}',
)

remote_support_agent = RemoteA2aAgent(
    name='support',
    description='Specialist agent for handling customer support queries',
    agent_card=f'http://localhost:10021{AGENT_CARD_WELL_KNOWN_PATH}',
)

# Router agent - uses SequentialAgent which automatically routes through sub-agents
# It will pass the user query to Data Agent first, then Support Agent
router_agent = SequentialAgent(
    name='router_agent',
    sub_agents=[remote_customer_data_agent, remote_support_agent],
)

router_agent_card = AgentCard(
    name='Router Agent',
    url='http://localhost:10022',
    description='Orchestrator agent that receives queries, analyzes intent, and routes to appropriate specialist agents',
    version='1.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'],
    default_output_modes=['text/plain'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[
        AgentSkill(
            id='route_query',
            name='Route Customer Query',
            description='Analyzes query intent and routes to appropriate specialist agent',
            tags=['routing', 'orchestration', 'coordination'],
            examples=['Get customer information for ID 5', 'I\'m customer 1 and need help']
        ),
        AgentSkill(
            id='coordinate_agents',
            name='Coordinate Multiple Agents',
            description='Coordinates responses from multiple specialist agents for complex queries',
            tags=['coordination', 'multi-agent', 'orchestration'],
            examples=['Update my email and show my ticket history']
        ),
        AgentSkill(
            id='analyze_intent',
            name='Analyze Query Intent',
            description='Analyzes customer queries to determine intent and required actions',
            tags=['analysis', 'intent', 'routing'],
            examples=['Determine if query needs data retrieval or support']
        ),
    ],
)