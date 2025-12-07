import asyncio
import httpx
from termcolor import colored
from a2a.types import AgentCard, TransportProtocol
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

# Target the Router
ROUTER_URL = "http://127.0.0.1:10022"

async def query_router(scenario, text):
    print(colored(f"\n{'='*80}", "blue"))
    print(colored(f"🎬 {scenario}", "green", attrs=["bold"]))
    print(colored(f"👤 Query: {text}", "cyan"))
    print(colored(f"{'='*80}", "blue"))
    
    # Increased timeout for multi-agent coordination
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 1. Fetch Router Card
            resp = await client.get(f"{ROUTER_URL}{AGENT_CARD_WELL_KNOWN_PATH}")
            if resp.status_code != 200:
                print(colored("❌ Router unavailable (Check agent_server.py)", "red"))
                return
            
            card = AgentCard(**resp.json())
            
            # 2. Setup Client
            factory = ClientFactory(ClientConfig(
                httpx_client=client, 
                supported_transports=[TransportProtocol.jsonrpc]
            ))
            a2a = factory.create(card)
            
            print(colored("⚡ Processing...\n", "yellow"))
            msg = create_text_message_object(content=text)
            
            # 3. Collect all streaming responses
            all_responses = []
            async for response in a2a.send_message(msg):
                if isinstance(response, tuple) and response:
                    try:
                        # Extract the text from the response artifact
                        answer = response[0].artifacts[0].parts[0].root.text
                        all_responses.append(answer)
                    except Exception as e:
                        # Silently skip unparseable responses
                        pass
            
            # 4. Display only the LAST response (final answer)
            if all_responses:
                final_response = all_responses[-1]
                
                # Show info if multiple responses were received
                if len(all_responses) > 1:
                    print(colored(f"ℹ️  Received {len(all_responses)} responses from agent coordination\n", "cyan"))
                
                print(colored("🤖 FINAL RESPONSE:", "green", attrs=["bold"]))
                print(colored(final_response, "white"))
            else:
                print(colored("⚠️ No text response received", "yellow"))

        except Exception as e:
            print(colored(f"❌ Error: {e}", "red"))
            import traceback
            traceback.print_exc()

async def main():
    """Run all test scenarios."""
    
    # Test Scenario 1: Simple Query
    await query_router(
        "Test 1: Simple Query",
        "Get customer information for ID 5"
    )
    await asyncio.sleep(1)
    
    # Test Scenario 2: Coordinated Query
    await query_router(
        "Test 2: Coordinated Query",
        "I'm customer 12345 and need help upgrading my account"
    )
    await asyncio.sleep(1)
    
    # Test Scenario 3: Complex Query
    await query_router(
        "Test 3: Complex Query",
        "Show me all active customers who have open tickets"
    )
    await asyncio.sleep(1)
    
    # Test Scenario 4: Escalation
    await query_router(
        "Test 4: Escalation",
        "I've been charged twice, please refund immediately!"
    )
    await asyncio.sleep(1)
    
    # Test Scenario 5: Multi-Intent
    await query_router(
        "Test 5: Multi-Intent Query",
        "I am customer 1. Update my email to newemail@example.com and show my ticket history"
    )
    
    print(colored(f"\n{'='*80}", "blue"))
    print(colored("✅ All tests completed!", "green", attrs=["bold"]))
    print(colored(f"{'='*80}", "blue"))

if __name__ == "__main__":
    asyncio.run(main())