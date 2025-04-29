import asyncio
import dataclasses # Import the dataclasses module
import os 
from dotenv import load_dotenv
import uuid # Import uuid for unique thread IDs

from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.runnables import RunnableConfig

# Assuming open_deep_research package is installed/available
# If running from workspace root and src is in PYTHONPATH, this might work:
try:
    from open_deep_research.graph import graph 
    from open_deep_research.state import ReportStateInput, ReportState, Section # Import Section
    from open_deep_research.configuration import Configuration, SearchAPI
except ImportError:
    print("Error importing open_deep_research. Make sure it's installed or src is in PYTHONPATH.")
    print("Try running 'pip install -e .' from the workspace root.")
    exit(1)
    
# Load environment variables from .env file
load_dotenv()

# --- Detailed Scenario Descriptions --- 
SCENARIOS = {
    "Scenario 1: Blue - US-Led Global AI Platforms": """
    Title: Continued US Tech Dominance Shaping Global AI & Web
    Description: A future largely reflecting current trends, where a handful of large US-based technology corporations continue to dominate cutting-edge AI development and deployment globally. Innovation is driven by intense competition between these players within a relatively deregulated international environment. This leads to a largely centralized web architecture, standardized AI platforms, and widespread adoption of US-developed services, potentially creating challenges for regional players and digital sovereignty.
    """,
    "Scenario 2: Green - Globally Coordinated AI for Societal Good": """
    Title: Regulated AI Focused on Global Challenges & Sustainability
    Description: A scenario characterized by significant international cooperation and regulatory frameworks (e.g., coordinated by bodies like the UN, EU, regional blocs) guiding AI development towards addressing major global issues like climate change, health, and sustainable development. The web landscape evolves with strong emphasis on ethical AI, data privacy, transparency, and measurable sustainability (e.g., carbon footprint tracking for digital services). Innovation in consumer-facing AI might be moderated by stricter compliance and ethical review processes.
    """,
    "Scenario 3: Yellow - Fragmented & Regionally Regulated AI": """
    Title: De-globalized Web with Strict Regional AI Governance
    Description: A future marked by increased digital fragmentation and de-globalization. Strong regional blocs (like the EU) implement comprehensive AI regulations driven by concerns over data sovereignty, security, and societal impact. This leads to data localization requirements, significant compliance overhead for businesses operating across regions, and potentially divergent web ecosystems. Public caution and regulatory hurdles may slow down the adoption of cutting-edge AI applications in some sectors.
    """,
    "Scenario 4: Red - Decentralized & Accelerated AI Disruption": """
    Title: Rapid, Decentralized AI Evolution Reshaping the Web
    Description: A scenario defined by explosive, decentralized AI innovation fueled by both powerful open-source models/communities and highly competitive corporate R&D in a minimally regulated global environment. Key characteristics include the rapid proliferation of specialized and autonomous AI agents, faster development cycles, the emergence of novel multimodal interfaces, and hyper-personalized web experiences. The web infrastructure itself adapts quickly, integrating more with edge computing and IoT. This creates tension between established centralized platforms and agile, decentralized actors, leading to a dynamic, perhaps chaotic, evolution towards highly autonomous systems and a fundamentally transformed web landscape.
    """
}

# Instantiate the checkpointer
checkpointer = MemorySaver()

async def run_report_generation(topic: str, scenario_details: str, report_structure: dict | str | None = None, output_filename: str | None = None):
    """
    Runs the research graph straight through to generate a report.
    """
    
    # --- Configuration --- 
    llm_provider = os.getenv("LLM_PROVIDER", "google_genai").lower()
    print(f"Using LLM Provider: {llm_provider}")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not set. Web search will fail.")

    # --- Define Model Mappings --- 
    planner_model = "gemini-1.5-pro-latest"
    writer_model = "gemini-1.5-flash-latest"
    if llm_provider == "openai":
        if not openai_api_key: print("Error: LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set."); exit(1)
        planner_model = "gpt-4o"
        writer_model = "gpt-4o-mini"
    elif llm_provider == "google_genai":
        if not google_api_key: print("Warning: LLM_PROVIDER is 'google_genai' but GOOGLE_API_KEY not set. Authentication might fail or use ADC.")
    else:
        print(f"Warning: Unsupported LLM_PROVIDER '{llm_provider}'. Defaulting to google_genai."); llm_provider = "google_genai"

    # --- Create Configuration Object --- 
    config_params = {
        "planner_provider": llm_provider, "planner_model": planner_model,
        "writer_provider": llm_provider, "writer_model": writer_model,
        "search_api": SearchAPI.TAVILY,
        "search_api_config": {"api_key": tavily_api_key} if tavily_api_key else {},
        "number_of_queries": 3, "max_search_depth": 2,
        "report_structure": report_structure or "Default report structure: Introduction, Main Body, Conclusion"
    }
    if llm_provider == "openai":
        config_params["planner_model_kwargs"] = {}
        config_params["writer_model_kwargs"] = {}
    config = Configuration(**config_params)
    
    # --- Prepare RunnableConfig --- 
    configurable_dict = dataclasses.asdict(config)
    if llm_provider == "google_genai":
        configurable_dict.pop("planner_model_kwargs", None)
        configurable_dict.pop("writer_model_kwargs", None)

    # Use a unique thread_id for each run 
    thread_id = str(uuid.uuid4()) 
    
    runnable_config = RunnableConfig(
        configurable=configurable_dict,
        checkpoint=checkpointer, # Include checkpointer for state persistence
        thread_id=thread_id 
    )

    # --- Input --- 
    initial_inputs = ReportStateInput(topic=topic, scenario_details=scenario_details)
    
    # --- Execute Graph --- 
    print(f"\n--- Starting Research Graph (Thread: {thread_id}) ---")
    final_report = None

    async for event in graph.astream_events(
        initial_inputs, runnable_config, version="v1", stream_mode="updates"
    ):
        kind = event["event"]
        name = event["name"]
        tags = event.get("tags", [])
        
        # --- Basic Event Logging --- 
        if kind == "on_tool_end":
            print(f"\n[{name}] Tool Output received.")
        elif kind == "on_llm_end":
            print(f"\n[{name}] LLM Call finished.")
        elif kind == "on_chain_end":
            if name != "LangGraph": print(f"\n[{name}] Finished.")
            # Check if the final report node completed
            if name == "compile_final_report": 
                # Get the output directly from the event data for this node
                output_data = event.get("data", {}).get("output")
                if isinstance(output_data, dict):
                     final_report = output_data.get("final_report", "Error: Final report not found in event data.")
                elif final_state_dict and final_state_dict.get("values"): # Fallback just in case
                     final_state = final_state_dict['values']
                     final_report = final_state.get("final_report", "Error: Final report not found in state.")
                else:
                     final_report = "Error: Could not retrieve final state from checkpointer or event."
                     
                print("\n--- FINAL REPORT --- ")
                print(final_report)
                if output_filename:
                    try:
                        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        with open(output_filename, "w") as f: f.write(final_report)
                        print(f"\nReport saved to {output_filename}")
                    except Exception as e:
                        print(f"\nError saving report: {e}") 
                else:
                    print("\n(Report not saved, no output_filename provided)")

                print("\nReport generation complete.")
                # No need to break here, stream will end naturally after END node
        else:
            # Log other events if needed for debugging
            # print(f"Event Kind: {kind}, Name: {name}")
            pass 
    
    print("\n--- Script Finished --- ")


# --- Example Usage ---
async def main():
    
    # Define a structure suitable for scenario analysis
    scenario_report_structure = {
        "sections": [
            {"name": "Scenario Narrative", "description": "Detailed description of the scenario, key characteristics, and timeline.", "research": True},
            {"name": "Key Drivers and Indicators", "description": "Underlying factors driving this scenario and measurable indicators to track its emergence.", "research": True},
            {"name": "Implications for Telia (Technical)", "description": "Specific technical challenges and opportunities for Telia's infrastructure, network, and internal systems.", "research": True},
            {"name": "Implications for Telia (Product Design)", "description": "Impact on Telia's B2C web product design, user experience, features, and required integrations.", "research": True},
            {"name": "Implications for Telia (Business Development)", "description": "Strategic business development considerations, potential partnerships, market positioning, and competitive responses for Telia.", "research": True},
            {"name": "Risks and Opportunities", "description": "A synthesized view of the major risks and opportunities presented to Telia by this scenario.", "research": True},
            {"name": "Scenario Summary", "description": "A concise executive summary of the scenario analysis and its core implications for Telia.", "research": False} # Uses context from researched sections
        ]
    }

    # Ensure the research directory exists
    output_dir = "research_telia_ai_web"
    os.makedirs(output_dir, exist_ok=True)

    # Determine which scenario to run (can be made dynamic, e.g., via command-line args)
    scenario_topic = "Scenario 1: Blue - US-Led Global AI Platforms"
    # scenario_topic = "Scenario 2: Green - Globally Coordinated AI for Societal Good"
    #scenario_topic = "Scenario 3: Yellow - Fragmented & Regionally Regulated AI"
    # scenario_topic = "Scenario 4: Red - Decentralized & Accelerated AI Disruption"
    
    # Get the corresponding detailed description
    scenario_details = SCENARIOS.get(scenario_topic)
    if not scenario_details:
        print(f"Error: Detailed description not found for topic: {scenario_topic}")
        exit(1)
        
    # Generate a filename based on the topic
    safe_topic_name = scenario_topic.lower().replace(" ", "_").replace(":", "").replace("/", "")
    output_filename = os.path.join(output_dir, f"{safe_topic_name}_report.md")


    print(f"--- Running Research for: {scenario_topic} ---")
    await run_report_generation(
        topic=scenario_topic,
        scenario_details=scenario_details, # Pass the detailed description
        report_structure=scenario_report_structure,
        output_filename=output_filename
    )
    

if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    # Handle potential nested loop errors if running in certain environments (like Jupyter)
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run nested event loop" in str(e):
            print("Detected nested event loop. Applying nest_asyncio.")
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        else:
             print(f"An unexpected runtime error occurred: {e}")
             raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e 








