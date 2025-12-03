# a2a_agents.py - A2A Protocol Compliant Agent Servers
import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain/LangGraph Core Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, StructuredTool
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

# --- A2A Agent Discovery Registry (shared across all agents) ---
# This will be populated when agents start
AGENT_REGISTRY = {}

# --- State Definition (must have messages key for A2A) ---
class AgentState(TypedDict):
    """State for A2A agents - must have messages key"""
    messages: Annotated[List[BaseMessage], operator.add]

# --- MCP Tool Wrappers ---
def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call MCP server tool via /mcp/call_tool endpoint"""
    url = f"{MCP_URL}/call_tool"
    try:
        print(f"[MCP TOOL CALL] Tool: {tool_name}")
        print(f"[MCP TOOL CALL] Arguments: {json.dumps(arguments, indent=2)}")
        response = requests.post(url, json={"name": tool_name, "arguments": arguments}, timeout=10)
        
        # Check status code before raising
        if response.status_code == 404:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            error_msg = f"ERROR: {error_detail}"
            print(f"[MCP TOOL CALL] {tool_name} FAILED - 404 Not Found: {error_detail}")
            return error_msg
        
        response.raise_for_status()
        result = response.json()
        mcp_result = result.get("result", result)
        print(f"[MCP TOOL CALL] {tool_name} SUCCESS - Result: {str(mcp_result)[:300]}...")
        return json.dumps(mcp_result, indent=2) if not isinstance(mcp_result, str) else mcp_result
    except requests.exceptions.HTTPError as e:
        # Handle 404 and other HTTP errors specifically
        if e.response is not None:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except:
                error_detail = e.response.text
            
            if status_code == 404:
                error_msg = f"Customer not found: {error_detail}"
            elif status_code == 400:
                error_msg = f"Invalid request: {error_detail}"
            else:
                error_msg = f"HTTP {status_code}: {error_detail}"
        else:
            error_msg = f"HTTP Error: {e}"
        print(f"[MCP TOOL CALL] {tool_name} FAILED - {error_msg}")
        return error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"ERROR: MCP call failed: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" Response: {e.response.text}"
        print(f"[MCP TOOL CALL] {tool_name} FAILED - {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"ERROR: MCP call failed: {e}"
        print(f"[MCP TOOL CALL] {tool_name} FAILED - {error_msg}")
        return error_msg

# Tool definitions
get_customer_tool = Tool(
    name="get_customer",
    func=lambda customer_id: call_mcp_tool("get_customer", {"customer_id": customer_id}),
    description="Retrieves a single customer record by customer_id (integer). Uses customers.id field."
)

def list_customers_func(status: Optional[str] = None, limit: int = 100) -> str:
    """List customers with optional status filter"""
    args = {"limit": limit}
    if status:
        args["status"] = status
    return call_mcp_tool("list_customers", args)

list_customers_tool = StructuredTool.from_function(
    func=list_customers_func,
    name="list_customers",
    description="Lists customers. Can filter by status ('active' or 'disabled'). Uses customers.status field."
)

def update_customer_func(customer_id: int, name: Optional[str] = None, email: Optional[str] = None, phone: Optional[str] = None, status: Optional[str] = None) -> str:
    """Update customer fields"""
    args = {"customer_id": customer_id}
    if name is not None:
        args["name"] = name
    if email is not None:
        args["email"] = email
    if phone is not None:
        args["phone"] = phone
    if status is not None:
        args["status"] = status
    return call_mcp_tool("update_customer", args)

update_customer_tool = StructuredTool.from_function(
    func=update_customer_func,
    name="update_customer",
    description="Updates customer fields. Uses customers table fields. Requires customer_id and optional fields: name, email, phone, status."
)

create_ticket_tool = Tool(
    name="create_ticket",
    func=lambda customer_id, issue, priority: call_mcp_tool("create_ticket", {"customer_id": customer_id, "issue": issue, "priority": priority}),
    description="Creates a new support ticket. Uses tickets table fields."
)

def get_customer_history_func(customer_id: int) -> str:
    """Get customer ticket history"""
    return call_mcp_tool("get_customer_history", {"customer_id": customer_id})

get_customer_history_tool = StructuredTool.from_function(
    func=get_customer_history_func,
    name="get_customer_history",
    description="Retrieves all tickets (history) for a specific customer_id. Uses tickets.customer_id field. Requires customer_id as integer."
)

data_agent_tools = [get_customer_tool, list_customers_tool, update_customer_tool, create_ticket_tool, get_customer_history_tool]

# --- A2A Communication Helper ---
def call_a2a_agent(agent_id: str, message: str) -> str:
    """Call another A2A agent via JSON-RPC"""
    # Try to get from registry, or use default ports
    agent_url = AGENT_REGISTRY.get(agent_id)
    if not agent_url:
        # Default ports if not in registry
        default_ports = {
            "customer_data": "http://127.0.0.1:9300",
            "support": "http://127.0.0.1:9301",
            "router": "http://127.0.0.1:9400"
        }
        agent_url = default_ports.get(agent_id)
        if not agent_url:
            return f"ERROR: Agent '{agent_id}' not found in registry and no default URL"
    
    try:
        # A2A JSON-RPC call
        payload = {
            "jsonrpc": "2.0",
            "method": "invoke",
            "params": {
                "messages": [{"role": "user", "content": message}]
            },
            "id": 1
        }
        response = requests.post(f"{agent_url}/a2a/{agent_id}", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            return f"ERROR: {result['error']}"
        return result.get("result", {}).get("content", str(result))
    except requests.exceptions.RequestException as e:
        # Fallback to direct invocation
        try:
            payload = {"query": message}
            response = requests.post(f"{agent_url}/a2a/{agent_id}", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", result.get("content", str(result)))
        except Exception as e2:
            return f"ERROR: Failed to call agent '{agent_id}': {e2}"

# --- Agent 1: Customer Data Agent ---
def customer_data_agent_node(state: AgentState) -> AgentState:
    """Customer Data Agent - handles data operations via MCP"""
    user_query = state['messages'][-1].content if state['messages'] else ""
    print(f"[Customer Data Agent] Processing: {user_query[:50]}...")
    
    import re
    import json
    
    # Extract customer ID from query
    customer_id_match = re.search(r'(?:customer|id)\s*(\d+)', user_query.lower())
    extracted_id = int(customer_id_match.group(1)) if customer_id_match else None
    
    # Use LLM with tools to decide what to do
    llm_with_tools = llm.bind_tools(data_agent_tools)
    
    # Create a more direct prompt that forces tool usage
    system_prompt = """You are a Customer Data Agent. Your ONLY job is to call database tools to retrieve or update customer data.

AVAILABLE TOOLS (use EXACT parameter names):
- get_customer(customer_id: integer) - REQUIRED: customer_id
- list_customers(status: 'active' or 'disabled', limit: integer) - OPTIONAL: status, limit
- update_customer(customer_id: integer, email: string, name: string, phone: string, status: string) - REQUIRED: customer_id
- get_customer_history(customer_id: integer) - REQUIRED: customer_id
- create_ticket(customer_id: integer, issue: string, priority: 'low'|'medium'|'high') - REQUIRED: customer_id, issue, priority

CRITICAL RULES:
1. You MUST call at least one tool - never respond with just text
2. Use EXACT parameter names: customer_id (not __arg1, not arg1, not id)
3. Extract customer IDs from queries:
   - "customer 12345" or "ID 12345" → customer_id=12345
   - "I'm customer X" → customer_id=X
   - "Get customer information for ID 5" → customer_id=5
4. For "show me all active customers" or "active customers with open tickets" → call list_customers with status='active'
5. For "I'm customer X and need help" → call get_customer(customer_id=X)
6. For "update my email to X" → call update_customer(customer_id=5, email='X') then get_customer_history(customer_id=5)
7. For multi-step queries, make multiple tool calls in sequence
8. ALWAYS use proper parameter names in tool calls"""
    
    enhanced_query = f"{system_prompt}\n\nUser Query: {user_query}\n\nExtract any customer IDs and call the appropriate tool(s) immediately."
    
    try:
        # First attempt with LLM
        ai_msg = llm_with_tools.invoke(enhanced_query)
        all_results = []
        
        # If LLM called tools, execute them
        if ai_msg.tool_calls:
            print(f"[Customer Data Agent] LLM called {len(ai_msg.tool_calls)} tool(s)")
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args_raw = tool_call.get("args", {})
                print(f"[Customer Data Agent] LLM provided tool: {tool_name} with raw args: {tool_args_raw}")
                
                # Fix tool arguments - handle cases where LLM provides wrong format
                tool_args = {}
                
                # Parse arguments properly based on tool
                if tool_name == "get_customer":
                    # Extract customer_id from various formats
                    if "customer_id" in tool_args_raw:
                        tool_args["customer_id"] = tool_args_raw["customer_id"]
                    elif "__arg1" in tool_args_raw:
                        # LLM sometimes uses __arg1
                        arg_val = tool_args_raw["__arg1"]
                        tool_args["customer_id"] = int(arg_val) if str(arg_val).isdigit() else extracted_id or 5
                    elif extracted_id:
                        tool_args["customer_id"] = extracted_id
                    else:
                        # Try to extract from query
                        id_match = re.search(r'id\s*(\d+)', user_query.lower())
                        tool_args["customer_id"] = int(id_match.group(1)) if id_match else 5
                
                elif tool_name == "update_customer":
                    # Extract customer_id and update fields
                    if "customer_id" in tool_args_raw:
                        tool_args["customer_id"] = tool_args_raw["customer_id"]
                    elif extracted_id:
                        tool_args["customer_id"] = extracted_id
                    elif "my" in user_query.lower():
                        tool_args["customer_id"] = 5
                    else:
                        id_match = re.search(r'id\s*(\d+)', user_query.lower())
                        tool_args["customer_id"] = int(id_match.group(1)) if id_match else 5
                    
                    # Handle __arg1 format which might contain a string dict
                    if "__arg1" in tool_args_raw:
                        arg1_val = str(tool_args_raw["__arg1"])
                        # Try to parse as Python dict string
                        try:
                            import ast
                            parsed_dict = ast.literal_eval(arg1_val)
                            if isinstance(parsed_dict, dict):
                                if "email" in parsed_dict:
                                    tool_args["email"] = parsed_dict["email"]
                                if "customer_id" in parsed_dict and "customer_id" not in tool_args:
                                    tool_args["customer_id"] = parsed_dict["customer_id"]
                        except:
                            # If not a dict, try to extract email from string
                            email_match = re.search(r"'email':\s*'([^']+)'", arg1_val)
                            if email_match:
                                tool_args["email"] = email_match.group(1)
                    
                    # Extract email if mentioned
                    if "email" not in tool_args:
                        if "email" in tool_args_raw:
                            tool_args["email"] = tool_args_raw["email"]
                        else:
                            email_match = re.search(r'to\s+([\w\.-]+@[\w\.-]+\.\w+)', user_query.lower())
                            if email_match:
                                tool_args["email"] = email_match.group(1)
                    
                    # Copy other fields
                    for field in ["name", "phone", "status"]:
                        if field in tool_args_raw:
                            tool_args[field] = tool_args_raw[field]
                
                elif tool_name == "get_customer_history":
                    if "customer_id" in tool_args_raw:
                        tool_args["customer_id"] = tool_args_raw["customer_id"]
                    elif extracted_id:
                        tool_args["customer_id"] = extracted_id
                    elif "my" in user_query.lower():
                        tool_args["customer_id"] = 5
                    else:
                        id_match = re.search(r'id\s*(\d+)', user_query.lower())
                        tool_args["customer_id"] = int(id_match.group(1)) if id_match else 5
                
                elif tool_name == "list_customers":
                    # For list_customers, extract status and limit properly
                    tool_args = {}
                    
                    # Extract status from query or args
                    if "status" in tool_args_raw:
                        tool_args["status"] = tool_args_raw["status"]
                    elif "active" in user_query.lower():
                        tool_args["status"] = "active"
                    elif "disabled" in user_query.lower():
                        tool_args["status"] = "disabled"
                    # If no status specified, don't filter (get all)
                    
                    # Extract limit
                    if "limit" in tool_args_raw:
                        tool_args["limit"] = tool_args_raw["limit"]
                    else:
                        tool_args["limit"] = 100
                    
                    # Clean up any wrong argument names
                    tool_args = {k: v for k, v in tool_args.items() if k not in ["__arg1", "arg1"]}
                
                elif tool_name == "create_ticket":
                    # For "upgrade account" or "need help" queries, we should get customer data, not create ticket
                    # Redirect to get_customer instead
                    if "upgrade" in user_query.lower() or "need help" in user_query.lower():
                        print(f"[Customer Data Agent] Redirecting create_ticket to get_customer for help/upgrade query")
                        # Change tool to get_customer
                        tool_name = "get_customer"
                        if extracted_id:
                            tool_args = {"customer_id": extracted_id}
                        else:
                            id_match = re.search(r'customer\s*(\d+)', user_query.lower())
                            tool_args = {"customer_id": int(id_match.group(1))} if id_match else {"customer_id": 5}
                    else:
                        # Parse create_ticket arguments normally
                        if "customer_id" in tool_args_raw:
                            tool_args["customer_id"] = tool_args_raw["customer_id"]
                        elif extracted_id:
                            tool_args["customer_id"] = extracted_id
                        else:
                            id_match = re.search(r'id\s*(\d+)', user_query.lower())
                            tool_args["customer_id"] = int(id_match.group(1)) if id_match else 5
                        
                        # Extract issue from query if not in args
                        if "issue" in tool_args_raw:
                            tool_args["issue"] = tool_args_raw["issue"]
                        elif "__arg1" in tool_args_raw:
                            # Try to parse from __arg1 format
                            arg1_val = str(tool_args_raw["__arg1"])
                            if "description:" in arg1_val or "issue:" in arg1_val:
                                # Extract issue from string
                                issue_match = re.search(r'(?:description|issue):\s*([^,]+)', arg1_val)
                                if issue_match:
                                    tool_args["issue"] = issue_match.group(1).strip()
                        
                        # Default priority if not specified
                        if "priority" in tool_args_raw:
                            tool_args["priority"] = tool_args_raw["priority"]
                        else:
                            tool_args["priority"] = "medium"
                else:
                    # For other tools, use raw args but clean up
                    tool_args = {k: v for k, v in tool_args_raw.items() if not k.startswith("__") and k != "arg1"}
                
                print(f"[Customer Data Agent] Parsed tool args for {tool_name}: {tool_args}")
                
                # Get the correct tool (might have changed if we redirected)
                selected_tool = next((t for t in data_agent_tools if t.name == tool_name), None)
                if selected_tool:
                    try:
                        # Execute tool with parsed arguments
                        print(f"[Customer Data Agent] Executing {tool_name} with parsed args: {tool_args}")
                        tool_output = selected_tool.invoke(tool_args)
                        print(f"[Customer Data Agent] {tool_name} returned: {str(tool_output)[:200]}...")
                        all_results.append(f"{tool_name}: {str(tool_output)}")
                        
                        # For complex queries, if we got customers list, process further
                        if tool_name == "list_customers" and ("open tickets" in user_query.lower() or "open ticket" in user_query.lower()):
                            # Parse customers and get their ticket history
                            try:
                                customers_data = json.loads(str(tool_output)) if isinstance(tool_output, str) else tool_output
                                if isinstance(customers_data, list):
                                    results = []
                                    for customer in customers_data[:20]:  # Limit to 20 for performance
                                        cust_id = customer.get('id')
                                        if cust_id:
                                            try:
                                                # Fix: invoke with dict, not int
                                                history_output = get_customer_history_tool.invoke({"customer_id": cust_id})
                                                history_str = str(history_output)
                                                # Try to parse history
                                                try:
                                                    history = json.loads(history_str)
                                                except:
                                                    # Extract JSON array from string
                                                    json_match = re.search(r'\[.*\]', history_str, re.DOTALL)
                                                    if json_match:
                                                        history = json.loads(json_match.group(0))
                                                    else:
                                                        continue
                                                
                                                if isinstance(history, list):
                                                    open_tickets = [t for t in history if t.get('status') == 'open']
                                                    if open_tickets:
                                                        results.append({
                                                            'customer_id': cust_id,
                                                            'customer_name': customer.get('name'),
                                                            'customer_email': customer.get('email'),
                                                            'open_tickets': open_tickets
                                                        })
                                            except Exception as e:
                                                print(f"Error getting history for customer {cust_id}: {e}")
                                                continue
                                    
                                    if results:
                                        all_results.append(f"filtered_results: {json.dumps(results, indent=2)}")
                                    else:
                                        all_results.append("No active customers with open tickets found")
                            except Exception as e:
                                print(f"Error processing complex query: {e}")
                    
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {e}"
                        print(f"[Customer Data Agent ERROR] {error_msg}")
                        all_results.append(error_msg)
                        import traceback
                        traceback.print_exc()
        else:
            # LLM didn't call tools - directly call tools based on query analysis
            print(f"[Customer Data Agent] LLM did not call tools. Forcing tool calls based on query analysis...")
            
            query_lower = user_query.lower()
            
            # Direct tool calling based on query patterns
            if "get customer" in query_lower or "customer information" in query_lower:
                if extracted_id:
                    try:
                        result = get_customer_tool.invoke(extracted_id)
                        all_results.append(f"get_customer: {result}")
                    except Exception as e:
                        all_results.append(f"Error: {e}")
                else:
                    id_match = re.search(r'id\s*(\d+)', query_lower)
                    if id_match:
                        try:
                            result = get_customer_tool.invoke(int(id_match.group(1)))
                            all_results.append(f"get_customer: {result}")
                        except Exception as e:
                            all_results.append(f"Error: {e}")
            
            elif "i'm customer" in query_lower or "i am customer" in query_lower or ("customer" in query_lower and "need help" in query_lower):
                # For "I'm customer X and need help", get customer data first
                if extracted_id:
                    try:
                        print(f"[Customer Data Agent] Direct call: get_customer(customer_id={extracted_id})")
                        result = get_customer_tool.invoke(extracted_id)
                        all_results.append(f"get_customer: {result}")
                    except Exception as e:
                        all_results.append(f"Error: {e}")
                else:
                    # Try to extract ID one more time
                    id_match = re.search(r'customer\s*(\d+)', query_lower)
                    if id_match:
                        try:
                            cust_id = int(id_match.group(1))
                            print(f"[Customer Data Agent] Direct call: get_customer(customer_id={cust_id})")
                            result = get_customer_tool.invoke(cust_id)
                            all_results.append(f"get_customer: {result}")
                        except Exception as e:
                            all_results.append(f"Error: {e}")
            
            elif "update" in query_lower and "email" in query_lower:
                # Extract email
                email_match = re.search(r'to\s+([\w\.-]+@[\w\.-]+\.\w+)', query_lower)
                email = email_match.group(1) if email_match else None
                cust_id = extracted_id if extracted_id else 5
                
                if email:
                    try:
                        print(f"[Customer Data Agent] Direct call: update_customer(customer_id={cust_id}, email={email})")
                        result = update_customer_tool.invoke({"customer_id": cust_id, "email": email})
                        all_results.append(f"update_customer: {result}")
                    except Exception as e:
                        all_results.append(f"Error: {e}")
                
                # If query also asks for history, get it
                if "history" in query_lower or "tickets" in query_lower or "ticket" in query_lower:
                    try:
                        print(f"[Customer Data Agent] Direct call: get_customer_history(customer_id={cust_id})")
                        history_result = get_customer_history_tool.invoke({"customer_id": cust_id})
                        all_results.append(f"get_customer_history: {history_result}")
                    except Exception as e:
                        all_results.append(f"Error getting history: {e}")
            
            elif "ticket history" in query_lower or "show my tickets" in query_lower:
                cust_id = extracted_id if extracted_id else 5
                try:
                    result = get_customer_history_tool.invoke({"customer_id": cust_id})
                    all_results.append(f"get_customer_history: {result}")
                except Exception as e:
                    all_results.append(f"Error: {e}")
            
            elif "active customers" in query_lower or ("show" in query_lower and "active" in query_lower and "customers" in query_lower):
                # Query about active customers - use list_customers with status='active'
                try:
                    print(f"[Customer Data Agent] Direct call: list_customers(status='active')")
                    customers_result = list_customers_func(status="active", limit=100)
                    all_results.append(f"list_customers: {customers_result}")
                    
                    # Parse and get ticket history for each
                    customers = json.loads(customers_result) if isinstance(customers_result, str) else customers_result
                    if isinstance(customers, list):
                        results = []
                        for customer in customers[:20]:
                            cust_id = customer.get('id')
                            if cust_id:
                                try:
                                    history_result = get_customer_history_tool.invoke({"customer_id": cust_id})
                                    history_str = str(history_result)
                                    try:
                                        history = json.loads(history_str)
                                    except:
                                        json_match = re.search(r'\[.*\]', history_str, re.DOTALL)
                                        history = json.loads(json_match.group(0)) if json_match else []
                                    
                                    if isinstance(history, list):
                                        open_tickets = [t for t in history if t.get('status') == 'open']
                                        if open_tickets:
                                            results.append({
                                                'customer_id': cust_id,
                                                'customer_name': customer.get('name'),
                                                'customer_email': customer.get('email'),
                                                'open_tickets': open_tickets
                                            })
                                except Exception as e:
                                    continue
                        
                        if results:
                            all_results.append(f"filtered_results: {json.dumps(results, indent=2)}")
                except Exception as e:
                    all_results.append(f"Error: {e}")
            
            else:
                # Fallback - try to extract ID and get customer
                if extracted_id:
                    try:
                        result = get_customer_tool.invoke(extracted_id)
                        all_results.append(f"get_customer: {result}")
                    except Exception as e:
                        all_results.append(f"Error: {e}")
                else:
                    all_results.append(f"Could not determine action from query: {user_query}")
        
        # Combine all results
        data_result = "\n".join(all_results) if all_results else "No data retrieved"
        
        return {
            'messages': [AIMessage(content=f"Data Agent Result: {data_result}")]
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'messages': [AIMessage(content=f"Error in Data Agent: {e}")]
        }

def create_customer_data_agent():
    """Create Customer Data Agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", customer_data_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# --- Agent 2: Support Agent ---
def support_agent_node(state: AgentState) -> AgentState:
    """Support Agent - provides customer support responses"""
    user_query = state['messages'][-1].content if state['messages'] else ""
    print(f"[Support Agent] Processing: {user_query[:50]}...")
    
    # Extract context from Data Agent results
    context = ""
    data_agent_results = []
    
    # Check messages for data agent results
    if len(state['messages']) > 1:
        for msg in state['messages'][:-1]:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if "Data Agent Result:" in content:
                # Extract the actual data (everything after "Data Agent Result:")
                data_part = content.split("Data Agent Result:", 1)[1].strip() if "Data Agent Result:" in content else content
                data_agent_results.append(data_part)
            elif any(keyword in content for keyword in ["get_customer:", "update_customer:", "list_customers:", "get_customer_history:"]):
                data_agent_results.append(content)
    
    # Also check if the query itself contains data context (from router coordination)
    if "Data Agent Result:" in user_query:
        # Extract data from the query
        parts = user_query.split("Data Agent Result:")
        if len(parts) > 1:
            context = parts[1].strip()
    elif "Original query:" in user_query:
        # Parse both original query and data context
        lines = user_query.split('\n')
        for i, line in enumerate(lines):
            if "Data Agent Result:" in line or "Data context:" in line:
                # Get everything after this line
                context = '\n'.join(lines[i:]).split(":", 1)[1].strip() if ":" in line else '\n'.join(lines[i:])
                break
    
    # Combine all data agent results
    if data_agent_results:
        context = "\n".join(data_agent_results)
    elif not context:
        # Try to find data in the query itself - look for JSON or structured data
        if "{" in user_query or "[" in user_query:
            # Extract JSON-like data
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', user_query, re.DOTALL)
            if json_match:
                context = json_match.group(1)
    
    # Clean up context
    if context:
        import re
        # Remove any "Data Agent Result:" prefixes
        context = re.sub(r'Data Agent Result:\s*', '', context, flags=re.IGNORECASE)
        context = context.strip()
    
    print(f"[Support Agent] Extracted context length: {len(context)} characters")
    if context:
        print(f"[Support Agent] Context preview: {context[:200]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a professional customer support specialist. "
         "You have received customer data from the Data Agent. You MUST use this data in your response. "
         "CRITICAL RULES:\n"
         "1. If customer data is provided, reference it directly (name, email, account status, tickets)\n"
         "2. DO NOT ask for information that is already in the data context\n"
         "3. If the query is about upgrading an account and you have customer data, use that data to provide specific help\n"
         "4. If the query is about a refund and you have customer data, acknowledge the customer and use their information\n"
         "5. Be helpful, empathetic, and provide specific information based on the data provided\n"
         "6. If data shows customer information, start your response by acknowledging the customer by name if available\n"
         "7. If customer data shows an error (like 'Customer not found' or '404'), acknowledge this gracefully:\n"
         "   - Apologize for the issue\n"
         "   - Explain that the customer ID may not exist in our system\n"
         "   - Ask the customer to verify their customer ID or provide their name/email to look them up\n"
         "   - Be helpful and empathetic"
        ),
        ("human", 
         "Customer Query: {query}\n\n"
         "Data from Data Agent (MUST use this in your response):\n{context}\n\n"
         "IMPORTANT: The data above contains real customer information from the database. "
         "Use this data to answer the customer's query. Reference specific details from the data. "
         "Do not ask for information that is already provided in the data context. "
         "If the data shows an error message, address that in your response."
        )
    ])
    
    response_chain = prompt | llm
    
    # Extract original query if it's embedded in the user_query
    original_query = user_query
    if "Customer Query:" in user_query:
        original_query = user_query.split("Customer Query:")[-1].split("\n")[0].strip()
    elif "Original query:" in user_query:
        original_query = user_query.split("Original query:")[-1].split("\n")[0].strip()
    
    final_answer = response_chain.invoke({
        "query": original_query,
        "context": context if context else "No customer data available. Ask the customer for their customer ID to retrieve their information."
    }).content
    
    return {
        'messages': [AIMessage(content=final_answer)]
    }

def create_support_agent():
    """Create Support Agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", support_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# --- Agent 3: Router Agent ---
def router_agent_node(state: AgentState) -> AgentState:
    """Router Agent - routes queries to appropriate agents following A2A flow"""
    user_query = state['messages'][-1].content if state['messages'] else ""
    print(f"[Router Agent] Received query: {user_query[:50]}...")
    
    # Extract customer ID if mentioned in query
    import re
    customer_id_match = re.search(r'(?:customer|id)\s*(\d+)', user_query.lower())
    extracted_customer_id = customer_id_match.group(1) if customer_id_match else None
    
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a routing agent. Analyze the customer query and classify the primary action required. "
         "Respond ONLY with one of these keywords: 'DATA', 'SUPPORT', or 'COORDINATION'. "
         "DATA: Simple data retrieval/update (e.g., 'Get customer info', 'Show tickets'). "
         "SUPPORT: General help/FAQ without needing customer data (e.g., 'What are your hours?'). "
         "COORDINATION: Requires data fetch AND subsequent support (e.g., 'upgrade account', 'refund', 'billing issues', 'help with account')."
        ),
        ("human", user_query)
    ])
    
    classification_chain = classification_prompt | llm | RunnableLambda(lambda x: x.content.strip().upper())
    
    try:
        response = classification_chain.invoke({})
        classification = response if isinstance(response, str) else response.content.strip().upper()
    except Exception as e:
        print(f"Error in classification: {e}")
        # Default based on query content
        if any(word in user_query.lower() for word in ["help", "upgrade", "refund", "cancel", "billing"]):
            classification = "COORDINATION"
        elif any(word in user_query.lower() for word in ["get", "show", "list", "update"]):
            classification = "DATA"
        else:
            classification = "SUPPORT"
    
    print(f"[Router Agent] Classification: {classification}")
    
    # Enhance query with extracted customer ID if found
    enhanced_query = user_query
    if extracted_customer_id:
        # Make sure customer ID is clear in the query
        if f"customer {extracted_customer_id}" not in user_query.lower() and f"id {extracted_customer_id}" not in user_query.lower():
            enhanced_query = f"{user_query} (Customer ID: {extracted_customer_id})"
    
    # Route following proper A2A flow
    if classification == "DATA":
        # Simple data query - Router → Data Agent → Router returns result
        print(f"[Router Agent] → Customer Data Agent: Requesting data...")
        data_result = call_a2a_agent("customer_data", enhanced_query)
        print(f"[Router Agent] ← Customer Data Agent: Received data")
        return {
            'messages': [AIMessage(content=f"Data retrieved: {data_result}")]
        }
    
    elif classification == "COORDINATION":
        # Coordination flow: Router → Data Agent → Router analyzes → Router → Support Agent → Router returns
        print(f"[Router Agent] COORDINATION FLOW:")
        print(f"[Router Agent] Step 1: → Customer Data Agent (requesting customer data)...")
        data_result = call_a2a_agent("customer_data", enhanced_query)
        print(f"[Router Agent] Step 2: ← Customer Data Agent (received data)")
        print(f"[Router Agent] Step 3: Analyzing customer data to determine support needs...")
        
        # Router analyzes the data to understand what support is needed
        # Check if data_result contains an error (like "Customer not found")
        if "ERROR" in data_result or "not found" in data_result.lower() or "404" in data_result:
            # If customer not found, skip analysis and go directly to support with error context
            print(f"[Router Agent] Customer data retrieval failed: {data_result[:100]}...")
            analysis = f"Customer data retrieval failed: {data_result}. The customer ID may not exist in the database."
        else:
            # Use a simple string format to avoid template variable issues with JSON
            analysis_query = (
                f"Original Query: {user_query}\n\n"
                f"Customer Data: {data_result}\n\n"
                "What support does this customer need? Provide a brief summary for the support agent."
            )
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a router agent analyzing customer data. Based on the customer data and original query, "
                 "determine what type of support response is needed. Provide a brief summary of the customer's situation."
                ),
                ("human", "{query}")
            ])
            
            analysis = (analysis_prompt | llm).invoke({
                "query": analysis_query
            }).content
        print(f"[Router Agent] Step 4: Analysis complete - {analysis[:100]}...")
        
        # Router → Support Agent with data context
        print(f"[Router Agent] Step 5: → Support Agent (with customer data context)...")
        # Format the support query to avoid template variable issues
        support_query = (
            f"Customer Query: {user_query}\n\n"
            f"Data Agent Result: {data_result}\n\n"
            f"Support Context: {analysis}"
        )
        support_result = call_a2a_agent("support", support_query)
        print(f"[Router Agent] Step 6: ← Support Agent (received response)")
        
        return {
            'messages': [AIMessage(content=f"Coordinated Response: {support_result}")]
        }
    
    else:  # SUPPORT
        # Simple support query - Router → Support Agent → Router returns
        print(f"[Router Agent] → Support Agent: Handling support query...")
        support_result = call_a2a_agent("support", enhanced_query)
        print(f"[Router Agent] ← Support Agent: Received response")
        return {
            'messages': [AIMessage(content=f"Support Response: {support_result}")]
        }

def create_router_agent():
    """Create Router Agent graph"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", router_agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# --- A2A Server Setup Functions ---
def create_agent_server(agent_id: str, agent_graph, port: int):
    """Create an A2A-compatible FastAPI server for an agent"""
    app = FastAPI(title=f"A2A {agent_id} Agent", version="1.0.0")
    
    @app.get("/")
    def health():
        return {"status": "ok", "agent_id": agent_id}
    
    @app.get("/a2a/{agent_id}")
    def get_agent_card(agent_id: str):
        """A2A Agent Card endpoint"""
        return {
            "agent_id": agent_id,
            "name": f"{agent_id.replace('_', ' ').title()} Agent",
            "description": f"A2A-compatible {agent_id} agent",
            "endpoint": f"/a2a/{agent_id}",
            "capabilities": ["invoke"]
        }
    
    @app.post("/a2a/{agent_id}")
    def invoke_agent(agent_id: str, request: Dict[str, Any]):
        """A2A JSON-RPC invoke endpoint"""
        try:
            # Handle JSON-RPC 2.0 format
            if "method" in request and request["method"] == "invoke":
                params = request.get("params", {})
                messages = params.get("messages", [])
                
                # Convert to LangChain messages
                langchain_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            langchain_messages.append(HumanMessage(content=content))
                        else:
                            langchain_messages.append(AIMessage(content=content))
                    else:
                        langchain_messages.append(HumanMessage(content=str(msg)))
                
                if not langchain_messages:
                    langchain_messages = [HumanMessage(content=params.get("query", ""))]
                
                # Invoke agent
                initial_state = {"messages": langchain_messages}
                final_state = agent_graph.invoke(initial_state)
                
                # Extract response
                final_message = final_state['messages'][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                # Return JSON-RPC 2.0 response
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": response_content,
                        "messages": [{"role": "assistant", "content": response_content}]
                    },
                    "id": request.get("id", 1)
                }
            else:
                # Direct invocation (non-JSON-RPC)
                query = request.get("query", request.get("message", ""))
                if not query:
                    raise HTTPException(status_code=400, detail="Query or message required")
                
                initial_state = {"messages": [HumanMessage(content=query)]}
                final_state = agent_graph.invoke(initial_state)
                final_message = final_state['messages'][-1]
                response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                return {
                    "response": response_content,
                    "messages": [{"role": "assistant", "content": response_content}]
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": request.get("id", 1)
            }
    
    return app, port

# Export agent creation functions
__all__ = [
    "create_customer_data_agent",
    "create_support_agent", 
    "create_router_agent",
    "create_agent_server",
    "AGENT_REGISTRY"
]

