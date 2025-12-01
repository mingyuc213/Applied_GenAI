# mcp_server.py (REST API VERSION)
import sqlite3
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from pyngrok import ngrok
from dotenv import load_dotenv

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCP_REST_Server")

# --- Configuration ---
DB_PATH = "support.db"
PORT = 8000

# Initialize FastAPI (Not FastMCP)
app = FastAPI(title="Customer Support REST Server")

# --- Database Helper ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Pydantic Models ---
class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None

class TicketCreate(BaseModel):
    customer_id: int
    issue: str
    priority: str

# --- API Endpoints (These match your agents.py calls) ---

@app.get("/mcp/get_customer/{customer_id}")
def get_customer(customer_id: int):
    conn = get_db_connection()
    customer = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    conn.close()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return dict(customer)

@app.get("/mcp/list_customers")
def list_customers(status: Optional[str] = None, limit: int = 10):
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
def update_customer(customer_id: int, data: CustomerUpdate):
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
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

@app.get("/mcp/create_ticket")
def create_ticket(customer_id: int, issue: str, priority: str):
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
def get_customer_history(customer_id: int):
    conn = get_db_connection()
    tickets = conn.execute("SELECT * FROM tickets WHERE customer_id = ?", (customer_id,)).fetchall()
    conn.close()
    return [dict(t) for t in tickets]

# --- Startup ---
if __name__ == "__main__":
    load_dotenv()
    
    # Setup Ngrok
    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
        try:
            public_url = ngrok.connect(PORT).public_url
            logger.info(f"Ngrok Tunnel Public URL: {public_url}")
            print(f"\n\n*** COPY THIS URL TO AGENTS.PY: {public_url}/mcp ***\n\n")
        except Exception as e:
            logger.error(f"Ngrok error: {e}")
    else:
        print("Warning: No NGROK_AUTHTOKEN found.")

    uvicorn.run(app, host="0.0.0.0", port=PORT)