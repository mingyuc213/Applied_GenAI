# a2a_agents.py - A2A Protocol Compliant Agent Servers using LangGraph built-in support
import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

# LangChain/LangGraph Core Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Configuration ---
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY
)

# --- State Definition (must have messages key for A2A) ---
class AgentState(TypedDict):
    """State for A2A agents - must have messages key"""
    messages: Annotated[List[BaseMessage], operator.add]

# --- Get MCP Tools using LangGraph's MCP Client ---
def get_mcp_tools():
    """Get MCP tools from server and convert to LangChain tools"""
    import requests
    import json
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    from typing import Optional
    
    try:
        # Get tools list from MCP server
        response = requests.get(f"{MCP_URL}/tools", timeout=5)
        response.raise_for_status()
        tools_data = response.json()
        
        langchain_tools = []
        
        for tool_def in tools_data.get("tools", []):
            tool_name = tool_def["name"]
            tool_desc = tool_def.get("description", "")
            input_schema = tool_def.get("inputSchema", {})
            
            def make_tool_func(name):
                def tool_func(**kwargs):
                    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    # Call MCP server
                    response = requests.post(
                        f"{MCP_URL}/call_tool",
                        json={"name": name, "arguments": filtered_kwargs},
                        timeout=10
                    )
                    if response.status_code == 404:
                        error_detail = response.json().get("detail", response.text)
                        return f"ERROR: {error_detail}"
                    response.raise_for_status()
                    result = response.json()
                    mcp_result = result.get("result", result)
                    return json.dumps(mcp_result, indent=2) if not isinstance(mcp_result, str) else mcp_result
                return tool_func
            
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            field_definitions = {}
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")
                if prop_type == "integer":
                    field_type = int
                elif prop_type == "boolean":
                    field_type = bool
                else:
                    field_type = str
                
                if "enum" in prop_schema:
                    field_type = str
                
                field_definitions[prop_name] = (
                    field_type if prop_name in required else Optional[field_type],
                    Field(
                        default=None if prop_name not in required else ...,
                        description=prop_schema.get("description", "")
                    )
                )
            
            ToolModel = create_model(f"{tool_name}Model", **field_definitions) if field_definitions else BaseModel
            
            tool_func = make_tool_func(tool_name)
            tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_desc,
                args_schema=ToolModel
            )
            langchain_tools.append(tool)
        
        return langchain_tools
    except Exception as e:
        print(f"Error getting MCP tools: {e}")
        return []

# Initialize tools
data_agent_tools = get_mcp_tools()

# --- A2A Communication Helper ---
def call_a2a_agent(agent_id: str, message: str) -> str:
    """Call another A2A agent via A2A protocol"""
    import requests
    
    # Default ports for agents
    default_ports = {
        "customer_data": 9300,
        "support": 9301,
        "router": 9400
    }
    
    port = default_ports.get(agent_id, 2024)
    url = f"http://127.0.0.1:{port}/a2a/{agent_id}"
    
    # Use invoke method (matches test client format)
    payload = {
        "jsonrpc": "2.0",
        "method": "invoke",
        "params": {
            "messages": [{"role": "user", "content": message}]
        },
        "id": 1
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Accept": "application/json"}, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            return f"ERROR: {result['error']}"
        # Extract content from response
        if "result" in result:
            if "content" in result["result"]:
                return result["result"]["content"]
            elif "artifacts" in result["result"]:
                artifacts = result["result"]["artifacts"]
                if artifacts and len(artifacts) > 0:
                    parts = artifacts[0].get("parts", [])
                    if parts and len(parts) > 0:
                        return parts[0].get("text", str(result))
        return str(result)
    except Exception as e:
        return f"ERROR: Failed to call agent '{agent_id}': {e}"

# --- Agent 1: Customer Data Agent ---
def customer_data_agent_node(state: AgentState) -> AgentState:
    """Customer Data Agent node - uses tools to retrieve/update customer data with multi-step support"""
    if not data_agent_tools:
        return {'messages': [AIMessage(content="ERROR: No MCP tools available")]}
    
    # Get the last user message
    user_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return {'messages': [AIMessage(content="ERROR: No user message found")]}
    
    user_query = user_messages[-1].content
    
    # Build conversation history (exclude the last message which we'll add separately)
    conversation_history = []
    for msg in state['messages'][:-1]:
        conversation_history.append(msg)
    
    # Enhanced system prompt with detailed instructions
    system_prompt = """You are a Customer Data Agent. Your job is to use MCP tools to retrieve or update customer data.

AVAILABLE TOOLS:
- get_customer(customer_id: int) - Get a single customer by ID
- list_customers(status: 'active'|'disabled', limit: int) - List customers, optionally filtered by status
- update_customer(customer_id: int, email: str, name: str, phone: str, status: str) - Update customer fields
- get_customer_history(customer_id: int) - Get all tickets for a customer
- create_ticket(customer_id: int, issue: str, priority: 'low'|'medium'|'high') - Create a support ticket

CRITICAL RULES:
1. ALWAYS call tools - never respond with just text without using tools
2. Extract customer IDs from queries:
   - "I'm customer 12345" → customer_id=12345
   - "Get customer information for ID 5" → customer_id=5
   - "Update my email" → if no ID specified, you may need to ask, but try to infer from context
3. For complex queries requiring multiple steps:
   - "Show me all active customers who have open tickets" → 
     Step 1: Call list_customers(status='active') to get all active customers
     Step 2: For each customer, call get_customer_history(customer_id) to get their tickets
     Step 3: Filter tickets where status='open' and return the results
   - Make multiple tool calls in sequence as needed
4. For multi-intent queries:
   - "Update my email to X and show my ticket history" →
     Step 1: Call update_customer(customer_id, email='X')
     Step 2: Call get_customer_history(customer_id)
     Step 3: Return both results
5. If a customer ID is mentioned (like "customer 12345"), use that ID even if the customer doesn't exist - the tool will return an error you can handle

You MUST call at least one tool. If you're unsure which tool to use, make your best guess based on the query."""
    
    llm_with_tools = llm.bind_tools(data_agent_tools)
    
    # Build messages for the LLM
    messages = [HumanMessage(content=system_prompt)] + conversation_history + [user_messages[-1]]
    
    # Iterative tool calling - continue until LLM doesn't call tools or gives final answer
    max_iterations = 10
    current_messages = messages
    
    for iteration in range(max_iterations):
        # Invoke LLM with current conversation
        ai_message = llm_with_tools.invoke(current_messages)
        current_messages.append(ai_message)
        
        # If LLM called tools, execute them
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            tool_messages = []
            
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                # Find and execute the tool
                tool = next((t for t in data_agent_tools if t.name == tool_name), None)
                if tool:
                    try:
                        tool_result = tool.invoke(tool_args)
                        tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call.get("id", "")))
                    except Exception as e:
                        tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call.get("id", "")))
            
            # Add tool results to conversation
            current_messages.extend(tool_messages)
            
            # Continue loop to let LLM process tool results and potentially make more calls
            continue
        else:
            # No more tool calls, LLM has final answer
            break
    
    # Get the final response (last AI message)
    final_response = current_messages[-1] if current_messages else ai_message
    
    # If final response is a tool call message, get a text response
    if hasattr(final_response, 'tool_calls') and final_response.tool_calls:
        # LLM made tool calls but we're out of iterations, get a summary
        final_response = llm.invoke(current_messages)
    
    return {'messages': [final_response]}

def create_customer_data_agent():
    """Create Customer Data Agent using LangGraph with MCP tools"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", customer_data_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# --- Agent 2: Support Agent ---
def support_agent_node(state: AgentState) -> AgentState:
    """Support Agent - provides customer support responses"""
    user_query = state['messages'][-1].content if state['messages'] else ""
    
    # Extract context from previous messages
    context = ""
    for msg in state['messages'][:-1]:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if "Data Agent Result:" in content:
            context = content.split("Data Agent Result:", 1)[1].strip()
            break
    
    if not context and "Data Agent Result:" in user_query:
        context = user_query.split("Data Agent Result:")[1].strip()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a professional customer support specialist. "
         "Use the customer data provided to answer queries. "
         "If customer data shows an error, acknowledge it gracefully and ask for verification."
        ),
        ("human", 
         "Customer Query: {query}\n\n"
         "Data from Data Agent: {context}"
        )
    ])
    
    original_query = user_query.split("Customer Query:")[-1].split("\n")[0].strip() if "Customer Query:" in user_query else user_query
    
    response = (prompt | llm).invoke({
        "query": original_query,
        "context": context or "No customer data available."
    })
    
    return {'messages': [AIMessage(content=response.content)]}

def create_support_agent():
    """Create Support Agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", support_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# --- Agent 3: Router Agent ---
def router_agent_node(state: AgentState) -> AgentState:
    """Router Agent - routes queries to appropriate agents"""
    user_query = state['messages'][-1].content if state['messages'] else ""
    
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Classify the query as 'DATA', 'SUPPORT', or 'COORDINATION'. "
         "DATA: Simple data retrieval/update queries (e.g., 'Get customer info', 'Show tickets', 'Update email'). "
         "SUPPORT: General help/FAQ without needing customer data (e.g., 'What are your hours?'). "
         "COORDINATION: Queries that need customer data AND a support response (e.g., 'I'm customer X and need help', 'upgrade account', 'refund', 'billing issues'). "
         "Also classify as COORDINATION if the query mentions a customer ID and asks for help/action."
        ),
        ("human", user_query)
    ])
    
    classification = (classification_prompt | llm | RunnableLambda(lambda x: x.content.strip().upper())).invoke({})
    
    # Fallback classification based on keywords
    if not classification or classification not in ["DATA", "SUPPORT", "COORDINATION"]:
        query_lower = user_query.lower()
        if any(word in query_lower for word in ["help", "upgrade", "refund", "cancel", "billing", "charged", "i'm customer", "i am customer"]):
            classification = "COORDINATION"
        elif any(word in query_lower for word in ["get", "show", "list", "update", "ticket history"]):
            classification = "DATA"
        else:
            classification = "SUPPORT"
    
    if classification == "DATA":
        data_result = call_a2a_agent("customer_data", user_query)
        return {'messages': [AIMessage(content=f"Data retrieved: {data_result}")]}
    
    elif classification == "COORDINATION":
        data_result = call_a2a_agent("customer_data", user_query)
        support_query = f"Customer Query: {user_query}\n\nData Agent Result: {data_result}"
        support_result = call_a2a_agent("support", support_query)
        return {'messages': [AIMessage(content=f"Coordinated Response: {support_result}")]}
    
    else:  # SUPPORT
        support_result = call_a2a_agent("support", user_query)
        return {'messages': [AIMessage(content=f"Support Response: {support_result}")]}

def create_router_agent():
    """Create Router Agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", router_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# Export agent creation functions
# Note: When deployed with langgraph dev, agents with messages key automatically expose /a2a/{assistant_id} endpoints
__all__ = [
    "create_customer_data_agent",
    "create_support_agent", 
    "create_router_agent",
]
