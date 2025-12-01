# agents.py (FINAL FIXED VERSION)
import os
import operator
import json
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Any, Optional

# 1. LOAD ENV VARS FIRST
load_dotenv()

# LangChain/LangGraph Core Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# Gemini LLM Integration
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration and Initialization ---
# Ensure you have GOOGLE_API_KEY in your .env file
if not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY not found in environment. Please check your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# The external endpoint for your MCP Server
MCP_URL = "https://homopolar-sally-childlike.ngrok-free.dev/mcp" 

# --- A. State Definition ---
class AgentState(TypedDict):
    """State for the LangGraph workflow."""
    messages: Annotated[List[BaseMessage], operator.add]
    context: str        # Data retrieved from the Data Agent
    next_node: str      # Used by the Router to decide where to go next

# --- B. MCP Tool Definitions ---

def call_mcp_tool(endpoint: str, params: Dict[str, Any] = None) -> str:
    """Helper to call the MCP server endpoints."""
    url = f"{MCP_URL}/{endpoint}"
    try:
        # Special handling for update_customer which expects a PUT body
        if params and 'data' in params and endpoint.startswith('update_customer'):
            # Convert Pydantic model to dict if needed, or use as is
            data_payload = params['data']
            if hasattr(data_payload, 'model_dump'):
                data_payload = data_payload.model_dump()
            response = requests.put(url, json=data_payload)
        elif params: 
            response = requests.get(url, params=params)
        else: 
            response = requests.get(url)
            
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.HTTPError as e:
        return f"ERROR: MCP call failed (Status {e.response.status_code}). Details: {e.response.text}"
    except Exception as e:
        return f"ERROR: Failed to connect to MCP server at {MCP_URL}. Is it running? {e}"

# --- C. Pydantic Schemas for Tools (THE FIX) ---
# Defining explicit classes resolves the PydanticUserError

class CustomerUpdateData(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None

class GetCustomerArgs(BaseModel):
    customer_id: int

class ListCustomersArgs(BaseModel):
    status: Optional[str] = None
    limit: int = 100

class UpdateCustomerArgs(BaseModel):
    customer_id: int
    data: CustomerUpdateData

class CreateTicketArgs(BaseModel):
    customer_id: int
    issue: str
    priority: str

class GetCustomerHistoryArgs(BaseModel):
    customer_id: int

# --- D. Tool Definitions ---

get_customer_tool = Tool(
    name="get_customer",
    func=lambda customer_id: call_mcp_tool(f"get_customer/{customer_id}"),
    description="Retrieves a single customer record by customer_id (integer).",
    args_schema=GetCustomerArgs,
)

list_customers_tool = Tool(
    name="list_customers",
    func=lambda status=None, limit=100: call_mcp_tool("list_customers", params={"status": status, "limit": limit}),
    description="Lists customers. Can filter by status ('active' or 'disabled').",
    args_schema=ListCustomersArgs,
)

update_customer_tool = StructuredTool.from_function(
    func=lambda customer_id, data: call_mcp_tool(f"update_customer/{customer_id}", params={"data": data}),
    name="update_customer",
    description="Updates customer fields.",
    args_schema=UpdateCustomerArgs,
)

create_ticket_tool = StructuredTool.from_function(
    func=lambda customer_id, issue, priority: call_mcp_tool("create_ticket", params={"customer_id": customer_id, "issue": issue, "priority": priority}),
    name="create_ticket",
    description="Creates a new support ticket.",
    args_schema=CreateTicketArgs, # Ensure CreateTicketArgs is defined correctly
)

get_customer_history_tool = Tool(
    name="get_customer_history",
    func=lambda customer_id: call_mcp_tool(f"get_customer_history/{customer_id}"),
    description="Retrieves all tickets (history) for a specific customer_id.",
    args_schema=GetCustomerHistoryArgs,
)

# List of all available tools
data_agent_tools = [get_customer_tool, list_customers_tool, update_customer_tool, create_ticket_tool, get_customer_history_tool]


# --- E. Agent Node Definitions ---

# --- E. 1. Router Agent Node ---
def router_node(state: AgentState) -> AgentState:
    """Analyzes intent using the LLM and sets the next destination."""
    user_query = state['messages'][-1].content
    print(f"\nLOG: Router received INITIAL query: {user_query}")
    
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a routing agent. Analyze the customer query and classify the primary action required. "
         "Respond ONLY with one of these keywords: 'DATA', 'SUPPORT', or 'COORDINATION'. "
         "DATA: Query is simple data retrieval/update (e.g., 'Get info', 'Update email')."
         "SUPPORT: Query is general help/FAQ (e.g., 'What are your hours?')."
         "COORDINATION: Query requires data fetch AND subsequent support (e.g., 'Cancel sub', 'Need help with ID 12345', 'Refund', 'Escalation')."
        ),
        ("human", user_query)
    ])
    
    classification_chain = classification_prompt | llm | RunnableLambda(lambda x: x.content.strip().upper())
    try:
        response = classification_chain.invoke({"user_query": user_query})
    except Exception as e:
        print(f"Error calling LLM: {e}")
        response = "SUPPORT" # Fallback

    if response in ("DATA", "COORDINATION"):
        destination = "customer_data_agent_node"
        print(f"LOG: Intent classified as '{response}'. Routing to Data Agent first.")
    else:
        destination = "support_agent_node"
        print(f"LOG: Intent classified as '{response}'. Routing directly to Support Agent.")

    return {"next_node": destination}


# --- E. 2. Customer Data Agent Node (MANUAL TOOL EXECUTION) ---
def customer_data_agent_node(state: AgentState) -> AgentState:
    """Uses LLM to select a tool and then manually executes it."""
    user_query = state['messages'][0].content 
    print(f"LOG: Customer Data Agent processing: '{user_query[:30]}...'")

    # 1. Bind tools. 
    # CRITICAL CHANGE: We add specific instructions to the query to encourage tool use.
    llm_with_tools = llm.bind_tools(data_agent_tools)
    
    # 2. Force the model to think about tools by appending a system hint
    # (Gemini sometimes chats if it thinks it's being helpful)
    enhanced_query = (
        f"User Query: {user_query}\n"
        "SYSTEM INSTRUCTION: You have access to database tools. "
        "If the user asks for a complex list (e.g., 'active customers with open tickets'), "
        "call 'list_customers' with status='active' as a first step. "
        "If the user provides an ID, call 'get_customer' or 'get_customer_history'. "
        "Do not ask for confirmation. Call the tool immediately."
    )
    
    try:
        ai_msg = llm_with_tools.invoke(enhanced_query)
    except Exception as e:
        print(f"Error invoking LLM with tools: {e}")
        return {
            'messages': [AIMessage(content="Error processing data request.")],
            'context': "Error",
            'next_node': 'support_agent_node'
        }
    
    data_result = "No data found."

    # 3. Check if the LLM wants to call a tool
    if ai_msg.tool_calls:
        # ... (Rest of the tool execution logic remains the same) ...
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"LOG: LLM selected tool: {tool_name} with args: {tool_args}")
            
            selected_tool = next((t for t in data_agent_tools if t.name == tool_name), None)
            
            if selected_tool:
                try:
                    tool_output = selected_tool.invoke(tool_args)
                    data_result = str(tool_output)
                    # Stop after first successful tool call for simplicity in this assignment
                    break 
                except Exception as e:
                    data_result = f"Error executing tool: {e}"
    else:
        # Fallback: If LLM chatted, we treat that chat as the "data result"
        # but we mark it so we know it didn't use a tool.
        data_result = f"NOTE: Agent did not use a tool. Response: {ai_msg.content}"

    if "ERROR" in data_result:
        print(f"ERROR: Tool Failure: {data_result}")
    
    new_messages = [AIMessage(content=f"Data fetched: {data_result}")]
    print(f"LOG: Data Agent finished. Context set.")

    return {
        'messages': new_messages,
        'context': data_result,
        'next_node': 'support_agent_node'
    }

    # 3. Check if the LLM wants to call a tool
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0] # Take the first tool call
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"LOG: LLM selected tool: {tool_name} with args: {tool_args}")
        
        # 4. Find and Execute the tool manually
        selected_tool = next((t for t in data_agent_tools if t.name == tool_name), None)
        
        if selected_tool:
            try:
                # Execute the tool!
                # Note: 'update_customer' args might need nesting check depending on how LLM returns it
                # The Pydantic model handles it, but let's be safe.
                tool_output = selected_tool.invoke(tool_args)
                data_result = str(tool_output)
            except Exception as e:
                data_result = f"Error executing tool: {e}"
        else:
            data_result = f"Error: Tool '{tool_name}' not found."
            
    else:
        # LLM didn't pick a tool, maybe it just chatted?
        data_result = ai_msg.content

    if "ERROR" in data_result:
        print(f"ERROR: Tool Failure: {data_result}")
    
    new_messages = [AIMessage(content=f"Data fetched: {data_result}")]
    print(f"LOG: Data Agent finished. Context set.")

    return {
        'messages': new_messages,
        'context': data_result,
        'next_node': 'support_agent_node'
    }


# --- E. 3. Support Agent Node ---
# agents.py (Fixed Support Agent Node)

def support_agent_node(state: AgentState) -> AgentState:
    """Provides a solution based on the user query and retrieved context."""
    original_query = state['messages'][0].content
    data_context = state['context']
    
    print(f"LOG: Support Agent combining query with data...")
    
    # 1. Define Prompt with VARIABLES (don't use f-string for data_context here)
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a professional customer support specialist. Synthesize the raw data/context into a clear response."
        ),
        ("human", 
         "Customer's original request: '{original_query}'.\n\n"
         "Retrieved data/context: {data_context}"  # <--- Use {variable} notation
        )
    ])

    # 2. Pass the ACTUAL values here
    response_chain = response_prompt | llm
    final_answer = response_chain.invoke({
        "original_query": original_query, 
        "data_context": data_context  # LangChain handles the curly braces safely here
    }).content

    print(f"LOG: Support Agent generated final answer.")
    
    return {
        'messages': [AIMessage(content=final_answer)],
        'next_node': 'end'
    }


# --- F. LangGraph Assembly ---
def assemble_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router_node", router_node)
    workflow.add_node("customer_data_agent_node", customer_data_agent_node)
    workflow.add_node("support_agent_node", support_agent_node)
    workflow.set_entry_point("router_node")
    
    workflow.add_conditional_edges("router_node", lambda x: x["next_node"],
        {"customer_data_agent_node": "customer_data_agent_node", "support_agent_node": "support_agent_node"}
    )
    workflow.add_edge("customer_data_agent_node", "support_agent_node")
    workflow.add_edge("support_agent_node", END)

    return workflow.compile()

app = assemble_graph()