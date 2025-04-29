from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic and scenario details
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]
    scenario_details = state.get("scenario_details", "") # Get scenario details
    # Feedback is no longer used here
    # feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # --- Initialize Writer Model --- 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_kwargs_from_config = get_config_value(configurable.writer_model_kwargs or {})
    
    # Conditionally pass model_kwargs based on provider
    if writer_provider == "google_genai":
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)
    else:
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_kwargs_from_config) 
    structured_llm_writer = writer_model.with_structured_output(Queries)
    # --- End Initialize Writer Model --- 

    # Format system instructions for query generation
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic, 
        scenario_details=scenario_details,
        report_organization=report_structure, 
        number_of_queries=number_of_queries
    )

    # Generate queries using the writer model
    results = await structured_llm_writer.ainvoke([SystemMessage(content=system_instructions_query),
                                               HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions for section planning
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, 
        scenario_details=scenario_details,
        report_organization=report_structure, 
        context=source_str, 
        feedback="N/A" # No feedback in this flow
    )

    # --- Initialize Planner Model --- 
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model_name = get_config_value(configurable.planner_model)
    planner_kwargs_from_config = get_config_value(configurable.planner_model_kwargs or {})

    # Special handling for claude-3.7-sonnet thinking budget (example)
    # Adapt this if using specific kwargs for other models/providers
    if planner_model_name == "claude-3-7-sonnet-latest":
         # This example assumes claude specific kwargs, adjust as needed
         # If google_genai planner needs specific args, add them here
         thinking_budget_kwargs = {"max_tokens": 20_000, "thinking": {"type": "enabled", "budget_tokens": 16_000}}
         # Combine with other kwargs if necessary
         final_planner_kwargs = {**planner_kwargs_from_config, **thinking_budget_kwargs} 
         planner_llm = init_chat_model(model=planner_model_name, 
                                       model_provider=planner_provider, 
                                       **final_planner_kwargs) # Pass specific kwargs directly
    # Conditionally pass model_kwargs based on provider
    elif planner_provider == "google_genai":
         # Pass only non-None specific kwargs if needed, or none if API key from env is enough
         # Example: planner_llm = init_chat_model(model=planner_model_name, model_provider=planner_provider, temperature=0.7) 
         planner_llm = init_chat_model(model=planner_model_name, model_provider=planner_provider)
    else:
         planner_llm = init_chat_model(model=planner_model_name, 
                                       model_provider=planner_provider,
                                       model_kwargs=planner_kwargs_from_config)
    # --- End Initialize Planner Model --- 

    # Report planner instructions
    planner_message = """Generate the sections of the report based on the topic and scenario details. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""
    
    # Generate the report sections
    structured_llm_planner = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm_planner.ainvoke([SystemMessage(content=system_instructions_sections),
                                                      HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}

async def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description, within the context of the scenario.
    
    Args:
        state: Current state containing section details and scenario details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    scenario_details = state.get("scenario_details", "") # Get scenario details
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # --- Initialize Writer Model --- 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_kwargs_from_config = get_config_value(configurable.writer_model_kwargs or {})

    # Conditionally pass model_kwargs based on provider
    if writer_provider == "google_genai":
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)
    else:
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_kwargs_from_config) 
    structured_llm_writer = writer_model.with_structured_output(Queries)
    # --- End Initialize Writer Model --- 

    # Format system instructions
    system_instructions = query_writer_instructions.format(
        topic=topic, 
        scenario_details=scenario_details,
        section_topic=section.description, 
        number_of_queries=number_of_queries
    )

    # Generate queries  
    queries = await structured_llm_writer.ainvoke([SystemMessage(content=system_instructions),
                                              HumanMessage(content="Generate search queries on the provided topic and scenario.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results, considering the scenario
    2. Evaluates the quality of the section based on the scenario
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results, section info, and scenario details
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    scenario_details = state.get("scenario_details", "") # Get scenario details
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions for section writing
    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic, 
        scenario_details=scenario_details,
        section_name=section.name, 
        section_topic=section.description, 
        context=source_str, 
        section_content=section.content
    )

    # --- Initialize Writer Model --- 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_kwargs_from_config = get_config_value(configurable.writer_model_kwargs or {})
    
    # Conditionally pass model_kwargs based on provider
    if writer_provider == "google_genai":
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)
    else:
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_kwargs_from_config)
    # --- End Initialize Writer Model --- 
    
    # Generate section content
    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # --- Initialize Planner/Reflection Model --- 
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model_name = get_config_value(configurable.planner_model)
    planner_kwargs_from_config = get_config_value(configurable.planner_model_kwargs or {})

    # Special handling for claude-3.7-sonnet thinking budget (example)
    if planner_model_name == "claude-3-7-sonnet-latest":
         thinking_budget_kwargs = {"max_tokens": 20_000, "thinking": {"type": "enabled", "budget_tokens": 16_000}}
         final_planner_kwargs = {**planner_kwargs_from_config, **thinking_budget_kwargs}
         reflection_model_llm = init_chat_model(model=planner_model_name, 
                                                model_provider=planner_provider, 
                                                **final_planner_kwargs).with_structured_output(Feedback)
    # Conditionally pass model_kwargs based on provider
    elif planner_provider == "google_genai":
         reflection_model_llm = init_chat_model(model=planner_model_name, 
                                                model_provider=planner_provider).with_structured_output(Feedback)
    else:
         reflection_model_llm = init_chat_model(model=planner_model_name, 
                                                model_provider=planner_provider,
                                                model_kwargs=planner_kwargs_from_config).with_structured_output(Feedback)
    # --- End Initialize Planner/Reflection Model --- 

    # Grade prompt 
    section_grader_message = ("Grade the report section based on the section topic and the overall scenario details. "
                              "Consider if the content adequately addresses the topic within the scenario context. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information relevant to the scenario.")
    
    # Format grader instructions
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic, 
        scenario_details=scenario_details,
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )
    
    # Generate feedback using the reflection model
    feedback = await reflection_model_llm.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                             HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research, considering the scenario context.
    
    Args:
        state: Current state with completed sections as context and scenario details
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    scenario_details = state.get("scenario_details", "") # Get scenario details
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(
        topic=topic, 
        scenario_details=scenario_details,
        section_name=section.name, 
        section_topic=section.description, 
        context=completed_report_sections
    )

    # --- Initialize Writer Model --- 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_kwargs_from_config = get_config_value(configurable.writer_model_kwargs or {})
    
    # Conditionally pass model_kwargs based on provider
    if writer_provider == "google_genai":
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)
    else:
         writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_kwargs_from_config) 
    # --- End Initialize Writer Model --- 
    
    # Generate section content
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources and scenario context.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        # Check if section content exists before assignment
        if section.name in completed_sections:
             section.content = completed_sections[section.name]
        else:
             # Handle cases where a section might not have been completed (e.g., skipped research)
             section.content = f"## {section.name}\n\nContent for this section was not generated."

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one, passing scenario details.
    
    Args:
        state: Current state with all sections, research context, and scenario details
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {
            "topic": state["topic"], 
            "scenario_details": state.get("scenario_details", ""),
            "section": s, 
            "report_sections_from_research": state["report_sections_from_research"]
        }) 
        for s in state["sections"] 
        if not s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")

# Conditional routing function after planning
def route_after_plan(state: ReportState):
    research_sections = [s for s in state["sections"] if s.research]
    if research_sections:
        print("--- Routing: Found research sections -> build_section_with_web_research ---")
        # Return Send commands to trigger research sub-graph for each research section
        return [ 
            Send(
                "build_section_with_web_research", 
                {
                    "topic": state["topic"], 
                    "scenario_details": state.get("scenario_details", ""), 
                    "section": s, 
                    "search_iterations": 0
                }
            ) 
            for s in research_sections
        ]
    else:
        print("--- Routing: No research sections -> gather_completed_sections ---")
        # If no research needed, skip to gathering step (which leads to final writing)
        return "gather_completed_sections"

# Add the conditional edge from generate_report_plan
builder.add_conditional_edges(
    "generate_report_plan", 
    route_after_plan, 
    # Map possible outcomes. Keys don't strictly matter here as the function
    # returns either a list of Sends (implicitly routing to the target node)
    # or the name of the next node.
    {
        "__default__": END, # Fallback, though route_after_plan should cover all cases
        "build_section_with_web_research": "build_section_with_web_research", # Explicit mapping if needed
        "gather_completed_sections": "gather_completed_sections" # Explicit mapping if needed
    } 
)

# Connect the output of the research sub-graph to gathering
builder.add_edge("build_section_with_web_research", "gather_completed_sections")

# Conditional routing from gathering to final writing (remains the same)
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])

# Connect final writing to final compilation
builder.add_edge("write_final_sections", "compile_final_report")

# End the graph after compilation
builder.add_edge("compile_final_report", END)

graph = builder.compile()
