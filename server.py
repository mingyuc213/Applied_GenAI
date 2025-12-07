from fastmcp import FastMCP
import sqlite3
import json
from typing import Optional, List, Dict, Any

# Initialize the MCP server
mcp = FastMCP("Customer Support System")

# Configuration
DB_PATH = "support.db"

def get_db_connection():
    """Creates and returns a database connection with dictionary cursor."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        return conn
    except sqlite3.Error as e:
        raise RuntimeError(f"Database connection failed: {e}")

@mcp.tool()
def list_available_tools() -> str:
    """
    Lists all available tools in this MCP server with their descriptions.
    """
    tools_info = [
        "1. get_customer(customer_id): Retrieve details for a specific customer.",
        "2. list_customers(status, limit): List customers filtered by status.",
        "3. update_customer(customer_id, data): Update customer fields.",
        "4. create_ticket(customer_id, issue, priority): Create a new support ticket.",
        "5. get_customer_history(customer_id): View all tickets for a customer.",
        "6. get_customers_with_open_tickets(): Find customers who have open tickets."
    ]
    return "Available Tools:\n" + "\n".join(tools_info)

@mcp.tool()
def get_customer(customer_id: int) -> str:
    """
    Retrieve details for a specific customer by their ID.
    Args:
        customer_id: The unique ID of the customer.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        row = cursor.fetchone()
        if row:
            return json.dumps(dict(row), indent=2)
        else:
            return f"Error: Customer with ID {customer_id} not found."
    finally:
        conn.close()

@mcp.tool()
def list_customers(status: Optional[str] = None, limit: int = 10) -> str:
    """
    List customers, optionally filtered by status.
    Args:
        status: Filter by 'active' or 'disabled'. If None, returns all.
        limit: Maximum number of results to return (default 10).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM customers"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return json.dumps([dict(row) for row in rows], indent=2)
    finally:
        conn.close()

@mcp.tool()
def update_customer(customer_id: int, data: str) -> str:
    """
    Update customer information.
    Args:
        customer_id: The ID of the customer to update.
        data: A JSON string containing the fields to update. 
              Valid keys: 'name', 'email', 'phone', 'status'.
    """
    try:
        update_fields = json.loads(data)
    except json.JSONDecodeError:
        return "Error: 'data' argument must be a valid JSON string."

    valid_keys = {'name', 'email', 'phone', 'status'}
    filtered_data = {k: v for k, v in update_fields.items() if k in valid_keys}
    
    if not filtered_data:
        return "Error: No valid fields provided for update."

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM customers WHERE id = ?", (customer_id,))
        if not cursor.fetchone():
            return f"Error: Customer with ID {customer_id} not found."

        set_clause = ", ".join([f"{key} = ?" for key in filtered_data.keys()])
        values = list(filtered_data.values())
        values.append(customer_id)
        
        cursor.execute(f"UPDATE customers SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return f"Successfully updated customer {customer_id}."
    except sqlite3.Error as e:
        return f"Database Error: {e}"
    finally:
        conn.close()

@mcp.tool()
def create_ticket(customer_id: int, issue: str, priority: str = "medium") -> str:
    """
    Create a new support ticket for a customer.
    Args:
        customer_id: The ID of the customer.
        issue: Description of the issue.
        priority: Priority level ('low', 'medium', 'high').
    """
    if priority not in ['low', 'medium', 'high']:
        return "Error: Priority must be 'low', 'medium', or 'high'."

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM customers WHERE id = ?", (customer_id,))
        if not cursor.fetchone():
            return f"Error: Customer with ID {customer_id} does not exist."

        cursor.execute(
            "INSERT INTO tickets (customer_id, issue, priority, status) VALUES (?, ?, ?, 'open')",
            (customer_id, issue, priority)
        )
        conn.commit()
        return f"Ticket created successfully. Ticket ID: {cursor.lastrowid}"
    except sqlite3.Error as e:
        return f"Database Error: {e}"
    finally:
        conn.close()

@mcp.tool()
def get_customer_history(customer_id: int) -> str:
    """
    Get the ticket history for a specific customer.
    Args:
        customer_id: The ID of the customer.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM customers WHERE id = ?", (customer_id,))
        customer_row = cursor.fetchone()
        if not customer_row:
            return f"Error: Customer with ID {customer_id} not found."
            
        cursor.execute("""
            SELECT id, issue, status, priority, created_at 
            FROM tickets 
            WHERE customer_id = ? 
            ORDER BY created_at DESC
        """, (customer_id,))
        
        tickets = [dict(row) for row in cursor.fetchall()]
        return json.dumps({
            "customer_id": customer_id,
            "customer_name": customer_row['name'],
            "total_tickets": len(tickets),
            "tickets": tickets
        }, indent=2)
    finally:
        conn.close()

@mcp.tool()
def get_customers_with_open_tickets() -> str:
    """
    Finds customers who have open tickets.
    Useful for generating reports or finding active issues.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT c.id, c.name, c.email, c.status
            FROM customers c
            JOIN tickets t ON c.id = t.customer_id
            WHERE t.status = 'open'
        """)
        rows = cursor.fetchall()
        return json.dumps([dict(row) for row in rows], indent=2)
    finally:
        conn.close()

if __name__ == "__main__":
    # Use 127.0.0.1 to match your agent configuration
    mcp.run(transport="sse", host="127.0.0.1", port=8000)