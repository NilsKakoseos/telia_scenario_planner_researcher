from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    scenario_details: str # Detailed description of the scenario

class ReportStateOutput(TypedDict):
    final_report: str # Final report

# Use a reducer lambda (lambda _curr, val: val) to take the last written value for non-accumulating fields
# This handles concurrent writes from parallel branches where the value should be the same.
LastValueReducer = lambda _curr, val: val

class ReportState(TypedDict):
    topic: Annotated[str, LastValueReducer] # Report topic    
    scenario_details: Annotated[str | None, LastValueReducer] # Detailed description of the scenario
    feedback_on_report_plan: Annotated[str | None, LastValueReducer] # Feedback on the report plan
    sections: Annotated[list[Section], LastValueReducer] # List of report sections 
    section: Annotated[Section | None, LastValueReducer] # Current section being processed (implicitly by parallel branches)
    search_iterations: Annotated[int | None, LastValueReducer] # Search iterations from sub-graph runs
    # Add search_queries key with reducer to handle implicit state merging
    search_queries: Annotated[list[SearchQuery] | None, LastValueReducer] # Search queries from sub-graph runs
    # Add source_str key with reducer to handle implicit state merging
    source_str: Annotated[str | None, LastValueReducer] # Last source string from sub-graph runs
    # completed_sections is designed to accumulate results from parallel branches
    completed_sections: Annotated[list, operator.add] # Send() API key 
    report_sections_from_research: Annotated[str | None, LastValueReducer] # String of any completed sections from research to write final sections
    final_report: Annotated[str | None, LastValueReducer] # Final report

class SectionState(TypedDict):
    topic: str # Report topic
    scenario_details: str | None = None # Detailed description of the scenario
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
