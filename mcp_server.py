# mcp_server.py - MCP Protocol Compliant Server
import sqlite3
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCP_Server")

# --- Configuration ---
DB_PATH = "support.db"
PORT = 8000

# Initialize FastAPI
app = FastAPI(title="MCP Customer Support Server", version="1.0.0")

# --- Database Helper ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- MCP Tool Definitions (JSON Schema compliant) ---
MCP_TOOLS = [
    {
        "name": "get_customer",
        "description": "Retrieves a single customer record by customer_id (integer). Uses customers.id field.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer ID to retrieve"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "list_customers",
        "description": "Lists customers. Can filter by status ('active' or 'disabled'). Uses customers.status field.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "disabled"],
                    "description": "Filter by customer status"
                },
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum number of customers to return"
                }
            }
        }
    },
    {
        "name": "update_customer",
        "description": "Updates customer fields. Uses customers table fields.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer ID to update"
                },
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["active", "disabled"]
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "create_ticket",
        "description": "Creates a new support ticket. Uses tickets table fields.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer ID for this ticket"
                },
                "issue": {
                    "type": "string",
                    "description": "Description of the issue"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Ticket priority level"
                }
            },
            "required": ["customer_id", "issue", "priority"]
        }
    },
    {
        "name": "get_customer_history",
        "description": "Retrieves all tickets (history) for a specific customer_id. Uses tickets.customer_id field.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer ID to get history for"
                }
            },
            "required": ["customer_id"]
        }
    }
]

# --- MCP Protocol Endpoints ---

@app.get("/mcp/tools")
def list_tools():
    """MCP: List available tools"""
    return {"tools": MCP_TOOLS}

@app.post("/mcp/call_tool")
def call_tool(request: Dict[str, Any]):
    """MCP: Call a tool by name with arguments"""
    tool_name = request.get("name")
    arguments = request.get("arguments", {})
    
    logger.info(f"MCP call_tool request: tool_name={tool_name}, arguments={arguments}")
    
    if tool_name == "get_customer":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id is required")
        conn = get_db_connection()
        customer = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        conn.close()
        if not customer:
            logger.warning(f"Customer {customer_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        logger.info(f"Retrieved customer {customer_id}: {dict(customer).get('name', 'N/A')}")
        return {"result": dict(customer)}
    
    elif tool_name == "list_customers":
        status = arguments.get("status")
        limit = arguments.get("limit", 100)
        conn = get_db_connection()
        query = "SELECT * FROM customers"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += f" LIMIT {limit}"
        customers = conn.execute(query, params).fetchall()
        conn.close()
        return {"result": [dict(c) for c in customers]}
    
    elif tool_name == "update_customer":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id is required")
        
        updates = {k: v for k, v in arguments.items() if k != "customer_id" and v is not None}
        if not updates:
            return {"result": {"message": "No fields to update"}}
        
        set_clauses = [f"{k} = ?" for k in updates.keys()]
        query = f"UPDATE customers SET {', '.join(set_clauses)} WHERE id = ?"
        params = list(updates.values()) + [customer_id]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        rows = cursor.rowcount
        conn.close()
        
        if rows == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        return {"result": {"status": "updated", "fields": updates}}
    
    elif tool_name == "create_ticket":
        customer_id = arguments.get("customer_id")
        issue = arguments.get("issue")
        priority = arguments.get("priority")
        
        if not all([customer_id, issue, priority]):
            raise HTTPException(status_code=400, detail="customer_id, issue, and priority are required")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO tickets (customer_id, issue, status, priority) VALUES (?, ?, 'open', ?)",
                (customer_id, issue, priority)
            )
            conn.commit()
            tid = cursor.lastrowid
            return {"result": {"status": "ticket_created", "ticket_id": tid}}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            conn.close()
    
    elif tool_name == "get_customer_history":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id is required")
        conn = get_db_connection()
        tickets = conn.execute("SELECT * FROM tickets WHERE customer_id = ?", (customer_id,)).fetchall()
        conn.close()
        logger.info(f"Retrieved {len(tickets)} tickets for customer {customer_id}")
        return {"result": [dict(t) for t in tickets]}
    
    else:
        logger.error(f"Unknown tool name requested: {tool_name}. Available tools: get_customer, list_customers, update_customer, create_ticket, get_customer_history")
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found. Available tools: get_customer, list_customers, update_customer, create_ticket, get_customer_history")

# --- Legacy REST endpoints for backward compatibility ---
@app.get("/mcp/get_customer/{customer_id}")
def get_customer_rest(customer_id: int):
    """Legacy REST endpoint"""
    conn = get_db_connection()
    customer = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    conn.close()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return dict(customer)

@app.get("/mcp/list_customers")
def list_customers_rest(status: Optional[str] = None, limit: int = 10):
    """Legacy REST endpoint"""
    conn = get_db_connection()
    query = "SELECT * FROM customers"
    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += f" LIMIT {limit}"
    customers = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(c) for c in customers]

@app.put("/mcp/update_customer/{customer_id}")
def update_customer_rest(customer_id: int, data: Dict[str, Any]):
    """Legacy REST endpoint"""
    updates = {k: v for k, v in data.items() if v is not None}
    if not updates:
        return {"message": "No fields to update"}
    
    set_clauses = [f"{k} = ?" for k in updates.keys()]
    query = f"UPDATE customers SET {', '.join(set_clauses)} WHERE id = ?"
    params = list(updates.values()) + [customer_id]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    rows = cursor.rowcount
    conn.close()
    
    if rows == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"status": "updated", "fields": updates}

@app.post("/mcp/create_ticket")
def create_ticket_rest(request: Dict[str, Any]):
    """Legacy REST endpoint"""
    customer_id = request.get("customer_id")
    issue = request.get("issue")
    priority = request.get("priority")
    
    if not all([customer_id, issue, priority]):
        raise HTTPException(status_code=400, detail="customer_id, issue, and priority are required")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO tickets (customer_id, issue, status, priority) VALUES (?, ?, 'open', ?)",
            (customer_id, issue, priority)
        )
        conn.commit()
        tid = cursor.lastrowid
        return {"status": "ticket_created", "ticket_id": tid}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/mcp/get_customer_history/{customer_id}")
def get_customer_history_rest(customer_id: int):
    """Legacy REST endpoint"""
    conn = get_db_connection()
    tickets = conn.execute("SELECT * FROM tickets WHERE customer_id = ?", (customer_id,)).fetchall()
    conn.close()
    return [dict(t) for t in tickets]

# --- Startup ---
if __name__ == "__main__":
    load_dotenv()
    logger.info(f"Starting MCP Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
