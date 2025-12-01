# test_client.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any

# Load the compiled graph from agents.py
from agents import app, AgentState

# Load environment variables (API Key)
load_dotenv()

# --- Configuration ---
# Your required test queries:
TEST_QUERIES = {
    "Scenario 1: Coordinated Query": "I'm customer 1 and need help upgrading my account",
    "Scenario 2: Escalation": "I've been charged twice (Customer ID 12345), please refund immediately!",
    "Scenario 3: Multi-Step": "Update my email to new@email.com and show my ticket history",
    "Simple Query": "Get customer information for ID 5",
    "Complex Query": "Show me all active customers who have open tickets",
}

def run_test_scenario(query: str, test_name: str) -> None:
    """
    Invokes the LangGraph application directly and prints the results.
    """
    print("=" * 70)
    print(f"Scenario: {test_name}")
    print(f"Query: {query}")
    print("-" * 70)

    initial_state = {"messages": [HumanMessage(content=query)]}

    try:
        final_state: AgentState = app.invoke(initial_state)

        print("✅ SUCCESS: Coordination Flow Complete")
        
        final_response = final_state['messages'][-1].content
        
        print("\n--- A2A Coordination Log (Flow Trace) ---")
        
        # Iterate through messages to trace the handoffs
        for i, msg in enumerate(final_state['messages']):
            if i == 0: continue
            
            content = msg.content
            
            # --- Identify Agent Role based on Content ---
            agent_role = "UNKNOWN"
            if "Data fetched:" in content:
                agent_role = "Customer Data Agent (A2A Handoff)"
                print(f"[1. {agent_role}]: {content}")
            
            elif i == len(final_state['messages']) - 1 and msg.type == 'ai':
                agent_role = "Support Agent (Final Response)"
                print(f"[2. {agent_role}]: See FINAL ANSWER below.")

        print("\n--- FINAL ANSWER ---")
        print(final_response)
        
    except Exception as e:
        print(f"❌ ERROR running scenario: {e}")

    print("=" * 70 + "\n")


def run_all_tests():
    """Runs all predefined test scenarios."""
    print("STARTING LANGGRAPH ORCHESTRATION TESTS (LLM INTEGRATED)\n")
    for name, query in TEST_QUERIES.items():
        run_test_scenario(query, name)
    
    print("ALL TESTS COMPLETE.")


if __name__ == "__main__":
    run_all_tests()