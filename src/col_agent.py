import os
import json
import traceback
from typing import Optional, List, override
from urllib.parse import urlencode

import dotenv
import instructor
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

# Load environment variables
dotenv.load_dotenv()

print("🔧 DEBUG: Loading environment variables...")
print(f"🔧 DEBUG: OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"🔧 DEBUG: OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")

# Simplified parameters - don't over-complicate this
class SearchParameters(BaseModel):
    query: str = Field(description="What to search for in Catalogue of Life")

# Data model for GPT to extract search terms
class CoLQueryParams(BaseModel):
    search_term: str = Field(..., description="Scientific or common name to search for")
    limit: Optional[int] = Field(5, description="Number of results (max 20)")

class CatalogueOfLifeAgent(IChatBioAgent):
    
    def __init__(self):
        print("🚀 DEBUG: Initializing CatalogueOfLifeAgent...")
        super().__init__()
        
        # Test OpenAI connection during initialization
        try:
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.instructor_client = instructor.patch(self.openai_client)
            print("✅ DEBUG: OpenAI client initialized successfully")
        except Exception as e:
            print(f"❌ DEBUG: Failed to initialize OpenAI client: {e}")
            raise
    
    @override
    def get_agent_card(self) -> AgentCard:
        print("🔧 DEBUG: Creating agent card...")
        
        card = AgentCard(
            name="Catalogue of Life Agent",
            description="Search for species and taxonomic information from the Catalogue of Life database",
            icon=None,  # No icon for now to avoid timeout issues
            url="http://localhost:29197",  # Use localhost for testing, change for production
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description="Search for species or taxonomic information",
                    parameters=SearchParameters
                )
            ]
        )
        
        print(f"✅ DEBUG: Agent card created - Name: {card.name}")
        print(f"✅ DEBUG: Agent URL: {card.url}")
        print(f"✅ DEBUG: Entrypoints: {[ep.id for ep in card.entrypoints]}")
        
        return card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: SearchParameters):
        print(f"\n🎯 DEBUG: Agent.run() called!")
        print(f"🎯 DEBUG: Entrypoint: {entrypoint}")
        print(f"🎯 DEBUG: Request: {request}")
        print(f"🎯 DEBUG: Params: {params}")
        
        # Validate entrypoint
        if entrypoint != "search":
            error_msg = f"Unknown entrypoint: {entrypoint}. Expected 'search'"
            print(f"❌ DEBUG: {error_msg}")
            await context.reply(error_msg)
            return

        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            process: IChatBioAgentProcess
            
            try:
                await process.log("🚀 Starting Catalogue of Life search")
                print("🔍 DEBUG: Starting search process...")
                
                # Step 1: Extract search parameters using GPT
                await process.log("🤖 Extracting search terms using GPT...")
                print("🤖 DEBUG: Calling GPT to extract search parameters...")
                
                query_instructions = """
                Extract a simple search term from the user's query for the Catalogue of Life database.
                Focus on:
                - Scientific names (like "Homo sapiens", "Panthera leo")
                - Common names (like "human", "lion", "tiger")
                - Genus names (like "Homo", "Panthera")
                
                Keep it simple - just extract the main species/organism they want to search for.
                """
                
                try:
                    query_params: CoLQueryParams = await self.instructor_client.chat.completions.create(
                        model="gpt-4o-mini",  # Use mini for faster/cheaper responses
                        messages=[
                            {"role": "system", "content": query_instructions},
                            {"role": "user", "content": f"Query: {request}\nParams: {params.query}"}
                        ],
                        response_model=CoLQueryParams,
                        temperature=0,
                        max_retries=2
                    )
                    
                    print(f"✅ DEBUG: GPT extracted - search_term: '{query_params.search_term}', limit: {query_params.limit}")
                    await process.log(f"✅ Extracted search term: '{query_params.search_term}'")
                    
                except Exception as gpt_error:
                    print(f"❌ DEBUG: GPT extraction failed: {gpt_error}")
                    await process.log(f"⚠️ GPT extraction failed, using fallback...")
                    
                    # Fallback: use the raw query
                    query_params = CoLQueryParams(
                        search_term=params.query or request,
                        limit=5
                    )
                    print(f"🔄 DEBUG: Using fallback search term: '{query_params.search_term}'")
                
                # Step 2: Query Catalogue of Life API
                await process.log(f"🌐 Querying Catalogue of Life for: '{query_params.search_term}'")
                print(f"🌐 DEBUG: Querying COL API...")
                
                col_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
                api_params = {
                    "q": query_params.search_term,
                    "limit": min(query_params.limit or 5, 20)  # Cap at 20
                }
                
                print(f"🔗 DEBUG: API URL: {col_url}")
                print(f"🔗 DEBUG: API Params: {api_params}")
                
                try:
                    response = requests.get(col_url, params=api_params, timeout=10)
                    print(f"📡 DEBUG: API Response Status: {response.status_code}")
                    print(f"📡 DEBUG: API Response URL: {response.url}")
                    
                    await process.log(f"📡 API responded with status: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_msg = f"Catalogue of Life API error: HTTP {response.status_code}"
                        print(f"❌ DEBUG: {error_msg}")
                        await process.log(f"❌ {error_msg}")
                        await context.reply(error_msg)
                        return
                    
                    # Parse JSON response
                    try:
                        data = response.json()
                        print(f"📊 DEBUG: Parsed JSON successfully")
                        print(f"📊 DEBUG: Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                    except json.JSONDecodeError as json_error:
                        print(f"❌ DEBUG: JSON decode error: {json_error}")
                        await process.log(f"❌ Failed to parse JSON response")
                        await context.reply("Error: Invalid response from Catalogue of Life API")
                        return
                    
                except requests.RequestException as req_error:
                    print(f"❌ DEBUG: Request error: {req_error}")
                    await process.log(f"❌ Network error: {str(req_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                
                # Step 3: Process results
                await process.log("📝 Processing search results...")
                print("📝 DEBUG: Processing results...")
                
                results = data.get("result", [])
                total = data.get("total", 0)
                
                print(f"📊 DEBUG: Found {len(results)} results out of {total} total")
                await process.log(f"📊 Found {len(results)} results (total: {total})")
                
                if len(results) == 0:
                    no_results_msg = f"No species found for '{query_params.search_term}' in Catalogue of Life"
                    print(f"🔍 DEBUG: {no_results_msg}")
                    await context.reply(no_results_msg)
                    return
                
                # Step 4: Format results
                await process.log("📋 Formatting results...")
                print("📋 DEBUG: Formatting results...")
                
                formatted_results = []
                reply_text = f"**Found {total} matches for '{query_params.search_term}' in Catalogue of Life:**\n\n"
                
                for i, item in enumerate(results[:5], 1):  # Show top 5
                    try:
                        print(f"🔍 DEBUG: Processing result {i}: {json.dumps(item, indent=2)[:200]}...")
                        
                        # Extract key information - adjust based on actual API response structure
                        scientific_name = item.get("scientificName", "Unknown")
                        rank = item.get("rank", "Unknown")
                        status = item.get("status", "Unknown")
                        
                        # Handle classification
                        classification = item.get("classification", [])
                        kingdom = None
                        phylum = None
                        
                        if classification:
                            for taxon in classification:
                                if taxon.get("rank") == "kingdom":
                                    kingdom = taxon.get("name")
                                elif taxon.get("rank") == "phylum":
                                    phylum = taxon.get("name")
                        
                        result_info = {
                            "scientificName": scientific_name,
                            "rank": rank,
                            "status": status,
                            "kingdom": kingdom,
                            "phylum": phylum
                        }
                        formatted_results.append(result_info)
                        
                        # Add to reply text
                        reply_text += f"**{i}. {scientific_name}**\n"
                        reply_text += f"   • Rank: {rank}\n"
                        reply_text += f"   • Status: {status}\n"
                        if kingdom:
                            reply_text += f"   • Kingdom: {kingdom}\n"
                        if phylum:
                            reply_text += f"   • Phylum: {phylum}\n"
                        reply_text += "\n"
                        
                    except Exception as item_error:
                        print(f"⚠️ DEBUG: Error processing result {i}: {item_error}")
                        await process.log(f"⚠️ Skipped malformed result {i}")
                        continue
                
                # Step 5: Create artifact
                await process.log("📎 Creating results artifact...")
                print("📎 DEBUG: Creating artifact...")
                
                artifact_data = {
                    "search_info": {
                        "query": query_params.search_term,
                        "total_found": total,
                        "showing": len(formatted_results),
                        "api_url": response.url
                    },
                    "results": formatted_results,
                    "raw_response": data  # Include raw data for debugging
                }
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"COL search results for '{query_params.search_term}'",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8')
                    )
                    print("✅ DEBUG: Artifact created successfully")
                    await process.log("✅ Artifact created with search results")
                    
                except Exception as artifact_error:
                    print(f"⚠️ DEBUG: Failed to create artifact: {artifact_error}")
                    await process.log(f"⚠️ Failed to create artifact: {str(artifact_error)}")
                
                # Step 6: Send final response
                await process.log("✅ Search completed successfully")
                print("✅ DEBUG: Sending final response to user")
                
                if len(formatted_results) < total:
                    reply_text += f"\n*Showing top {len(formatted_results)} of {total} total results. See artifact for complete data.*"
                
                await context.reply(reply_text)
                print("🎉 DEBUG: Response sent successfully!")
                
            except Exception as e:
                error_msg = f"Unexpected error during search: {str(e)}"
                print(f"💥 DEBUG: MAJOR ERROR: {error_msg}")
                print(f"💥 DEBUG: Traceback: {traceback.format_exc()}")
                
                await process.log(f"❌ Error: {error_msg}")
                await context.reply(f"Sorry, an error occurred: {str(e)}")


# Test the agent locally before running the server
def test_agent():
    """Test function to verify agent works before starting server"""
    print("\n🧪 TESTING AGENT LOCALLY...")
    
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        print(f"✅ TEST: Agent card created successfully")
        print(f"✅ TEST: Agent name: {card.name}")
        print(f"✅ TEST: Entrypoints: {[ep.id for ep in card.entrypoints]}")
        return True
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        print(f"❌ TRACEBACK: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Catalogue of Life Agent...")
    
    # Test the agent first
    if not test_agent():
        print("❌ Agent test failed! Fix errors before starting server.")
        exit(1)
    
    print("✅ Agent test passed! Starting server...")
    
    try:
        from ichatbio.server import run_agent_server
        
        agent = CatalogueOfLifeAgent()
        
        print("📍 Server starting on: http://localhost:9999")
        print("🔗 Agent card will be at: http://localhost:9999/.well-known/agent.json")
        print("🔗 Test the API manually: https://api.checklistbank.org/dataset/3LR/nameusage/search?q=tiger")
        print("Press Ctrl+C to stop")
        
        run_agent_server(agent, host="0.0.0.0", port=9999)
        
    except Exception as e:
        print(f"💥 SERVER ERROR: {e}")
        print(f"💥 TRACEBACK: {traceback.format_exc()}")