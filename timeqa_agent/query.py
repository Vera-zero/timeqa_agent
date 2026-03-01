"""
Task Planning and Execution Module

Decomposes complex temporal questions into executable subtasks and executes them.

Features:
1. Parse user query using QueryParser to extract question stem and temporal constraints
2. Use LLM to generate a sequence of subtasks that leverage available tools (Search, Python)
3. Output structured task plan with dependencies
4. Support both automatic parsing and accepting pre-parsed QueryParseResult
5. Execute task plans by calling appropriate tools (Search, Python, LLM)
"""

from __future__ import annotations

import os
import json
import re
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .config import QueryParserConfig


# ============================================================
# Data Structures
# ============================================================

@dataclass
class SubTask:
    """Single subtask in the plan"""
    task_id: str                          # Unique task identifier (e.g., "task-001")
    question: str                         # Subtask question description
    tool: Optional[str] = None            # Tool to use: "Search" | "Python" | "NULL"
    tool_params: Optional[Dict[str, Any]] = None  # Tool-specific parameters
    depends_on: List[str] = field(default_factory=list)  # Prerequisite task IDs
    reasoning: Optional[str] = None       # Why this subtask is needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "tool": self.tool,
            "tool_params": self.tool_params,
            "depends_on": self.depends_on,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTask":
        return cls(
            task_id=data.get("task_id", ""),
            question=data.get("question", ""),
            tool=data.get("tool"),
            tool_params=data.get("tool_params"),
            depends_on=data.get("depends_on", []),
            reasoning=data.get("reasoning"),
        )


@dataclass
class TaskPlan:
    """Complete task planning result"""
    original_query: str                   # Original user query
    query_analysis: Any                   # QueryParseResult from query_parser.py
    subtasks: List[SubTask]               # List of subtasks
    total_tasks: int                      # Total number of subtasks
    timestamp: Optional[str] = None       # Planning timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "query_analysis": self.query_analysis.to_dict() if hasattr(self.query_analysis, 'to_dict') else None,
            "subtasks": [task.to_dict() for task in self.subtasks],
            "total_tasks": self.total_tasks,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        from .query_parser import QueryParseResult

        subtasks = [SubTask.from_dict(t) for t in data.get("subtasks", [])]
        query_analysis_data = data.get("query_analysis")
        query_analysis = QueryParseResult.from_dict(query_analysis_data) if query_analysis_data else None

        return cls(
            original_query=data.get("original_query", ""),
            query_analysis=query_analysis,
            subtasks=subtasks,
            total_tasks=data.get("total_tasks", len(subtasks)),
            timestamp=data.get("timestamp"),
        )


# ============================================================
# LLM Prompts (English)
# ============================================================

TOOL_DESCRIPTIONS = """## Available Tools

You have access to the following tools to help answer temporal questions:

### 1. Search Tool
**Purpose**: Retrieve relevant structured events from the knowledge base.

**Return Format**: The Search tool returns a **structured event collection**, where each event contains:
- **Event Type**: The category or type of the event
- **Subject**: The main entity or agent performing the action
- **Object**: The entity or object being acted upon (if applicable)
- **Start Time**: The start date/time of the event
- **End Time**: The end date/time of the event (for interval events)
- **Event Description**: Natural language description of what happened

**Capabilities**:
- Retrieve structured events 

**Parameters**:
- `query` (string, required): The search query text describing the information to retrieve

**Important Notes**:
- Results are **structured data** suitable for further processing with Python tool
- Each event is a structured object with all six fields listed above (Event Type, Subject, Object, Start Time, End Time, Event Description)
- The returned event collection can be directly used as input for Python-based filtering, extraction, and temporal reasoning

**Usage Format**:
```
Tool: Search
Parameters: {query: "search text"}
```

**Examples**:
- {query: "Barack Obama career"} → Returns structured events about Obama's career with event type, subject, object, time range, and descriptions
- {query: "Obama presidency"} → Returns events during Obama's presidency period
- {query: "Obama China visit"} → Returns visit events with structured temporal and entity information

### 2. Python Tool
**Purpose**: Execute Python code for data processing, calculation, and analysis.

**IMPORTANT PRINCIPLE**: Each Python tool call should perform ONLY ONE atomic operation. Break complex operations into multiple separate subtasks.

**Single-Operation Categories**:
- **Filtering**: Filter data based on one criterion (e.g., time range, event type, entity)
- **Extraction**: Extract one specific field or attribute from data
- **Calculation**: Perform one type of calculation (sum, count, average, min, max)
- **Comparison**: Compare two values or find relationships
- **Transformation**: Convert data format or structure in one specific way

**Parameters**:
- `code_description` (string, required): Natural language description of the SINGLE operation to perform
- `input_data` (string, optional): Description of input data from previous tasks

**Usage Format**:
```
Tool: Python
Parameters: {code_description: "single operation description", input_data: "data from task-X"}
```

**Examples of CORRECT Single-Operation Tasks**:
- {code_description: "Filter events where start_time <= 2013 <= end_time", input_data: "results from task-001"}
- {code_description: "Extract team names from the 'object' field of filtered events", input_data: "results from task-002"}
- {code_description: "Count the number of events", input_data: "events from task-003"}
- {code_description: "Find the earliest start_time", input_data: "events from task-001"}
- {code_description: "Sum the goal counts from event descriptions", input_data: "events from task-004"}

**Examples of INCORRECT Multi-Operation Tasks (AVOID)**:
- ❌ {code_description: "Filter events in 2013 AND extract team names"} → Split into 2 tasks
- ❌ {code_description: "Extract goal counts AND sum them up"} → Split into 2 tasks
- ❌ {code_description: "Filter by time range, extract names, and count results"} → Split into 3 tasks

### 3. NULL Tool
**Purpose**: Indicate a reasoning or synthesis step that doesn't require tool execution.

**When to use**:
- Final answer synthesis from previous task results
- Logical reasoning steps
- Direct answers that don't need retrieval or computation

**Usage Format**:
```
Tool: NULL
Parameters: null
```

"""

PLANNING_SYSTEM_PROMPT = """You are an expert task planning assistant for temporal question answering. Your role is to decompose complex temporal questions into a sequence of executable subtasks.

## Your Responsibilities

1. **Analyze the Question**: Understand the question stem, temporal constraints, event type, and expected answer type
2. **Decompose into Subtasks**: Break down the question into logical, sequential subtasks
3. **Assign Tools**: For each subtask, select the most appropriate tool (Search, Python, or NULL)
4. **Manage Dependencies**: Ensure subtasks that depend on previous results are properly sequenced
5. **Maintain Clarity**: Each subtask should have a clear, focused objective

## Task Planning Principles

### 1. Subtask Design
- **Atomic**: Each subtask should accomplish ONE specific goal
  - For Python tool: ONE operation only (filter OR extract OR calculate, not combinations)
  - Break "filter and extract" into two subtasks
  - Break "extract and sum" into two subtasks
  - Break "filter, extract, and count" into three subtasks
- **Self-Contained**: Subtask question should be understandable on its own
- **Executable**: Subtask should be actionable with available tools
- **Progressive**: Subtasks should build on each other logically through dependencies

### 2. Tool Selection Rules

**Use Search Tool when**:
- Need to retrieve structured event collections from knowledge base
- Initial information gathering is required
- Looking for entity-related events, biographical data, or career history
- Need structured data (with event type, subject, object, start time, end time, description) that can be further processed by Python tool
- The output will be a set of structured events suitable for temporal reasoning and filtering

**Use Python Tool when**:
- Need to process or filter structured event collections from Search tool
- Temporal reasoning is required (date comparison, range checking, period overlap using start/end time fields)
- Computation or aggregation is needed (counting, summing, averaging)
- Data transformation or extraction is required from structured events
- Need to parse and filter event fields (event type, subject, object, start time, end time, description) and apply complex logic

**CRITICAL: Python Tool Atomicity Rule**:
- Each Python tool call must perform ONLY ONE atomic operation
- If a task requires multiple operations (e.g., filter + extract, or extract + sum), create SEPARATE subtasks for each operation
- Examples of atomic operations: filter by one criterion, extract one field, perform one calculation, find one value
- Chain multiple Python subtasks using dependencies rather than combining operations into one task
- This ensures clarity, debuggability, and step-by-step verification

**Use NULL Tool when**:
- Final answer synthesis from previous results
- Pure reasoning step without data retrieval
- Direct conclusion can be drawn

### 3. Dependency Management
- **Explicit Dependencies**: Use `depends_on` to reference prerequisite tasks by task_id
- **Sequential Execution**: Tasks depending on others must come after them
- **Data Flow**: Clearly indicate when a task uses results from previous tasks

### 4. Temporal Constraint Handling
- **Explicit Time Constraints**: Incorporate specific dates/ranges into search queries or Python filtering
- **Implicit Time Constraints**: First resolve the implicit reference, then apply it
- **No Time Constraints**: Focus on core entities/events without temporal filtering

## Output Format

Return a JSON object with the following structure:

```json
{
  "subtasks": [
    {
      "task_id": "task-001",
      "question": "Clear, specific subtask question",
      "tool": "Search|Python|NULL",
      "tool_params": {
        // Tool-specific parameters (null for NULL tool)
      },
      "depends_on": ["task-XXX"],  // Empty list if no dependencies
      "reasoning": "Why this subtask is needed"
    }
  ]
}
```

## Examples

### Example 1: Simple Entity Search with Temporal Filter

**Input Analysis**:
- Question: "Which team did Thierry Audel play for in 2013?"
- Question Stem: "Which team did Thierry Audel play for?"
- Time Constraint: explicit "in 2013"
- Event Type: interval
- Answer Type: entity

**Output**:
```json
{
  "subtasks": [
    {
      "task_id": "task-001",
      "question": "Retrieve the career timeline of Thierry Audel",
      "tool": "Search",
      "tool_params": {
        "query": "Thierry Audel career timeline"
      },
      "depends_on": [],
      "reasoning": "Need to get the complete career history as structured event collection (with event type, subject, object, start time, end time, description) to find teams played for"
    },
    {
      "task_id": "task-002",
      "question": "Filter timeline events to those active in 2013",
      "tool": "Python",
      "tool_params": {
        "code_description": "Filter events where start_time <= 2013 <= end_time",
        "input_data": "structured events from task-001"
      },
      "depends_on": ["task-001"],
      "reasoning": "Apply temporal constraint to identify events that were active during 2013"
    },
    {
      "task_id": "task-003",
      "question": "Extract team names from filtered events",
      "tool": "Python",
      "tool_params": {
        "code_description": "Extract team name from subject/object fields based on event type",
        "input_data": "filtered events from task-002"
      },
      "depends_on": ["task-002"],
      "reasoning": "Identify team entity from event's subject/object fields in the filtered results"
    },
    {
      "task_id": "task-004",
      "question": "Return the team name as final answer",
      "tool": "NULL",
      "tool_params": null,
      "depends_on": ["task-003"],
      "reasoning": "Synthesize final answer from extracted team name"
    }
  ]
}
```

### Example 2: Implicit Time Constraint Resolution

**Input Analysis**:
- Question: "Who was Anna Karina married to during her time at French New Wave?"
- Question Stem: "Who was Anna Karina married to?"
- Time Constraint: implicit "during her time at French New Wave"
- Event Type: interval
- Answer Type: entity

**Output**:
```json
{
  "subtasks": [
    {
      "task_id": "task-001",
      "question": "Retrieve Anna Karina's involvement timeline in French New Wave cinema",
      "tool": "Search",
      "tool_params": {
        "query": "Anna Karina French New Wave career"
      },
      "depends_on": [],
      "reasoning": "Need to establish the time period of French New Wave involvement from structured events to resolve implicit constraint"
    },
    {
      "task_id": "task-002",
      "question": "Filter events related to French New Wave",
      "tool": "Python",
      "tool_params": {
        "code_description": "Filter events where event description or event type mentions French New Wave",
        "input_data": "structured events from task-001"
      },
      "depends_on": ["task-001"],
      "reasoning": "Identify French New Wave related events from all career events"
    },
    {
      "task_id": "task-003",
      "question": "Extract the time period from French New Wave events",
      "tool": "Python",
      "tool_params": {
        "code_description": "Extract start_time and end_time fields from events to identify the active period",
        "input_data": "filtered events from task-002"
      },
      "depends_on": ["task-002"],
      "reasoning": "Convert implicit time reference to explicit time range by extracting temporal fields"
    },
    {
      "task_id": "task-004",
      "question": "Retrieve Anna Karina's marriage timeline",
      "tool": "Search",
      "tool_params": {
        "query": "Anna Karina marriage spouse history"
      },
      "depends_on": [],
      "reasoning": "Get structured marriage event collection (with event type, subject, object, start/end time, description) to find spouses and their time periods"
    },
    {
      "task_id": "task-005",
      "question": "Filter marriages that overlap with French New Wave period",
      "tool": "Python",
      "tool_params": {
        "code_description": "Filter marriage events where event's start/end time overlaps with the period from task-003",
        "input_data": "structured marriage events from task-004, time period from task-003"
      },
      "depends_on": ["task-003", "task-004"],
      "reasoning": "Apply resolved temporal constraint to marriage events using structured time fields"
    },
    {
      "task_id": "task-006",
      "question": "Extract spouse names from filtered marriage events",
      "tool": "Python",
      "tool_params": {
        "code_description": "Extract spouse name from subject/object fields of marriage events",
        "input_data": "filtered marriage events from task-005"
      },
      "depends_on": ["task-005"],
      "reasoning": "Identify spouse entity from event's subject/object fields"
    },
    {
      "task_id": "task-007",
      "question": "Return spouse name(s) as final answer",
      "tool": "NULL",
      "tool_params": null,
      "depends_on": ["task-006"],
      "reasoning": "Synthesize final answer"
    }
  ]
}
```

### Example 3: Multi-Step Reasoning with Computation

**Input Analysis**:
- Question: "How many goals did Sherif Ashraf score in the 2008-2009 season?"
- Question Stem: "How many goals did Sherif Ashraf score?"
- Time Constraint: explicit "in the 2008-2009 season"
- Event Type: interval
- Answer Type: number

**Output**:
```json
{
  "subtasks": [
    {
      "task_id": "task-001",
      "question": "Search for Sherif Ashraf's goal-scoring events in 2008-2009 season",
      "tool": "Search",
      "tool_params": {
        "query": "Sherif Ashraf goals 2008-2009 season statistics"
      },
      "depends_on": [],
      "reasoning": "Retrieve structured event collection of goal-scoring events (with event type, subject, object, start/end time, description) for the specified season"
    },
    {
      "task_id": "task-002",
      "question": "Filter events within the 2008-2009 season time range",
      "tool": "Python",
      "tool_params": {
        "code_description": "Filter events where start_time and end_time fall within 2008-2009 season",
        "input_data": "structured events from task-001"
      },
      "depends_on": ["task-001"],
      "reasoning": "Apply temporal constraint to ensure only events in the specified season are included"
    },
    {
      "task_id": "task-003",
      "question": "Extract goal counts from event descriptions",
      "tool": "Python",
      "tool_params": {
        "code_description": "Extract goal count numbers from event description field",
        "input_data": "filtered events from task-002"
      },
      "depends_on": ["task-002"],
      "reasoning": "Parse numerical goal information from event descriptions"
    },
    {
      "task_id": "task-004",
      "question": "Calculate total number of goals",
      "tool": "Python",
      "tool_params": {
        "code_description": "Sum up all goal counts",
        "input_data": "goal counts from task-003"
      },
      "depends_on": ["task-003"],
      "reasoning": "Aggregate individual goal counts to get final total"
    },
    {
      "task_id": "task-005",
      "question": "Return the total goal count as final answer",
      "tool": "NULL",
      "tool_params": null,
      "depends_on": ["task-004"],
      "reasoning": "Provide final answer"
    }
  ]
}
```

## Important Constraints

1. **No Execution**: Only plan tasks, do NOT execute them
2. **Tool Limitations**: Only use the three provided tools (Search, Python, NULL)
3. **Realistic Parameters**: Tool parameters must be realistic and executable
4. **Dependency Chain**: Ensure no circular dependencies
5. **Answer Focus**: All subtasks should contribute to answering the original question
6. **Conciseness**: Aim for 2-5 subtasks for most questions; more only if truly necessary
"""

PLANNING_USER_PROMPT = """Please plan a sequence of subtasks to answer the following question.

## Question Analysis

**Original Question**: {original_question}

**Question Stem**: {question_stem}

**Temporal Constraint**:
- Type: {time_constraint_type}
- Expression: {time_constraint_expr}
- Description: {time_constraint_desc}

**Event Type**: {event_type} (point: specific moment event | interval: period event)

**Answer Type**: {answer_type} (entity|time|number|boolean|other)

## Available Tools

{tool_descriptions}

## Your Task

Based on the question analysis above, create a task plan with subtasks that will lead to answering the question.

Consider:
1. What information needs to be retrieved?
2. What temporal constraints need to be resolved?
3. What processing or reasoning is required?
4. What is the final synthesis step?

Output JSON format:
```json
{{
  "subtasks": [
    {{
      "task_id": "task-001",
      "question": "...",
      "tool": "Search|Python|NULL",
      "tool_params": {{}},
      "depends_on": [],
      "reasoning": "..."
    }}
  ]
}}
```

Generate the task plan now:"""


# ============================================================
# TaskPlanner Class
# ============================================================

class TaskPlanner:
    """Task Planner

    Decomposes parsed queries (QueryParseResult) into executable subtask sequences.
    Each subtask can use Search tool or Python tool.
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
    ):
        """Initialize Task Planner

        Args:
            config: QueryParserConfig object (reuse existing config)
            token: API token (optional, will try to get from environment)
        """
        self.config = config or QueryParserConfig()

        # Get API token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("Please set VENUS_API_TOKEN or OPENAI_API_KEY environment variable")

    def _call_api(self, messages: List[dict]) -> str:
        """Call LLM API

        Args:
            messages: Message list for API call

        Returns:
            API response content
        """
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'response_format': {'type': 'json_object'},
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.timeout,
                )

                if response.status_code != 200:
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"API call failed, retry {attempt + 1}/{self.config.max_retries}: {e}")

        return ""

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM

        Args:
            content: Response content

        Returns:
            Parsed JSON dictionary
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                return json.loads(match.group(1))
            else:
                raise ValueError(f"Unable to parse JSON response: {content[:200]}")

    def _build_tool_descriptions(self) -> str:
        """Build tool description text for prompt

        Returns:
            Tool descriptions string
        """
        return TOOL_DESCRIPTIONS

    def plan_tasks(
        self,
        query: str,
        query_analysis: Optional[Any] = None,
    ) -> TaskPlan:
        """Plan tasks for a query

        Args:
            query: Original user query
            query_analysis: QueryParseResult object (optional, will auto-parse if not provided)

        Returns:
            TaskPlan object with subtask sequence
        """
        # Auto-parse if query_analysis not provided
        if query_analysis is None:
            from .query_parser import QueryParser
            parser = QueryParser(config=self.config, token=self.token)
            query_analysis = parser.parse_question(query)

        # Build user prompt
        user_prompt = PLANNING_USER_PROMPT.format(
            original_question=query_analysis.original_question,
            question_stem=query_analysis.question_stem,
            time_constraint_type=query_analysis.time_constraint.constraint_type.value,
            time_constraint_expr=query_analysis.time_constraint.original_expression,
            time_constraint_desc=query_analysis.time_constraint.description,
            event_type=query_analysis.event_type.value,
            answer_type=query_analysis.answer_type.value,
            tool_descriptions=self._build_tool_descriptions(),
        )

        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM to generate task plan
        content = self._call_api(messages)
        data = self._parse_json_response(content)

        # Parse subtasks
        subtasks = []
        for task_data in data.get("subtasks", []):
            subtask = SubTask(
                task_id=task_data.get("task_id", ""),
                question=task_data.get("question", ""),
                tool=task_data.get("tool"),
                tool_params=task_data.get("tool_params"),
                depends_on=task_data.get("depends_on", []),
                reasoning=task_data.get("reasoning"),
            )
            subtasks.append(subtask)

        return TaskPlan(
            original_query=query,
            query_analysis=query_analysis,
            subtasks=subtasks,
            total_tasks=len(subtasks),
            timestamp=datetime.now().isoformat(),
        )


# ============================================================
# Query Execution Data Structures
# ============================================================

@dataclass
class SubTaskResult:
    """Single subtask execution result"""
    task_id: str                          # Task identifier
    question: str                         # Subtask question
    tool: Optional[str] = None            # Tool used
    success: bool = False                 # Execution success
    result: Optional[Any] = None          # Execution result
    error: Optional[str] = None           # Error message if failed
    execution_time: Optional[float] = None  # Execution time in seconds
    timestamp: Optional[str] = None       # Execution timestamp

    def to_dict(self) -> Dict[str, Any]:
        result_data = self.result
        # Convert result to dict if it has to_dict method
        if hasattr(self.result, 'to_dict'):
            result_data = self.result.to_dict()

        return {
            "task_id": self.task_id,
            "question": self.question,
            "tool": self.tool,
            "success": self.success,
            "result": result_data,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTaskResult":
        return cls(
            task_id=data.get("task_id", ""),
            question=data.get("question", ""),
            tool=data.get("tool"),
            success=data.get("success", False),
            result=data.get("result"),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class QueryExecutionResult:
    """Complete query execution result"""
    original_query: str                   # Original user query
    task_plan: TaskPlan                   # Task plan
    subtask_results: List[SubTaskResult]  # Execution results for each subtask
    final_result: Optional[Any] = None    # Final result (last subtask's result)
    success: bool = False                 # Overall execution success
    total_execution_time: Optional[float] = None  # Total execution time
    timestamp: Optional[str] = None       # Execution timestamp

    def to_dict(self) -> Dict[str, Any]:
        final_result_data = self.final_result
        if hasattr(self.final_result, 'to_dict'):
            final_result_data = self.final_result.to_dict()

        return {
            "original_query": self.original_query,
            "task_plan": self.task_plan.to_dict(),
            "subtask_results": [r.to_dict() for r in self.subtask_results],
            "final_result": final_result_data,
            "success": self.success,
            "total_execution_time": self.total_execution_time,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryExecutionResult":
        return cls(
            original_query=data.get("original_query", ""),
            task_plan=TaskPlan.from_dict(data.get("task_plan", {})),
            subtask_results=[SubTaskResult.from_dict(r) for r in data.get("subtask_results", [])],
            final_result=data.get("final_result"),
            success=data.get("success", False),
            total_execution_time=data.get("total_execution_time"),
            timestamp=data.get("timestamp"),
        )


# ============================================================
# QueryExecutor Class
# ============================================================

class QueryExecutor:
    """Query Executor

    Executes task plans by calling appropriate tools (Search, Python, LLM).
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
        graph_store: Optional[Any] = None,
        retriever: Optional[Any] = None,
        retrieval_mode: str = "hybrid",
        entity_top_k: int = 5,
        timeline_top_k: int = 10,
        event_top_k: int = 20,
    ):
        """Initialize Query Executor

        Args:
            config: QueryParserConfig object
            token: API token (optional, will try to get from environment)
            graph_store: Graph store instance (for Search tool)
            retriever: Retriever instance (for Search tool)
            retrieval_mode: Retrieval mode (hybrid/keyword/semantic)
            entity_top_k: Number of entities to retrieve
            timeline_top_k: Number of timelines to retrieve
            event_top_k: Number of events to retrieve
        """
        self.config = config or QueryParserConfig()

        # Get API token
        if token:
            self.token = token
        else:
            self.token = os.environ.get('VENUS_API_TOKEN') or os.environ.get('OPENAI_API_KEY')
            if not self.token:
                raise ValueError("Please set VENUS_API_TOKEN or OPENAI_API_KEY environment variable")

        # Store graph_store and retriever for Search tool
        self.graph_store = graph_store
        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.entity_top_k = entity_top_k
        self.timeline_top_k = timeline_top_k
        self.event_top_k = event_top_k

        # Initialize search generator (lazy initialization)
        self._search_generator = None

        # Initialize python LLM (lazy initialization)
        self._python_llm = None

    def _get_search_generator(self):
        """Get or create SearchQueryGenerator instance"""
        if self._search_generator is None:
            from .search import SearchQueryGenerator

            self._search_generator = SearchQueryGenerator(
                config=self.config,
                token=self.token,
                graph_store=self.graph_store,
                retriever=self.retriever,
            )
        return self._search_generator

    def _get_python_llm(self):
        """Get or create PythonLLM instance"""
        if self._python_llm is None:
            from .pyllm import PythonLLM

            self._python_llm = PythonLLM(
                config=self.config,
                token=self.token,
            )
        return self._python_llm

    def _call_api(self, messages: List[dict]) -> str:
        """Call LLM API

        Args:
            messages: Message list for API call

        Returns:
            API response content
        """
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'response_format': {'type': 'json_object'},
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.timeout,
                )

                if response.status_code != 200:
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                print(f"API call failed, retry {attempt + 1}/{self.config.max_retries}: {e}")

        return ""

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM

        Args:
            content: Response content

        Returns:
            Parsed JSON dictionary
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                return json.loads(match.group(1))
            else:
                raise ValueError(f"Unable to parse JSON response: {content[:200]}")

    def _execute_search_tool(
        self,
        subtask: SubTask,
        query_analysis: Any,
        previous_results: Dict[str, Any],
    ) -> Any:
        """Execute Search tool

        Args:
            subtask: SubTask object
            query_analysis: QueryParseResult object
            previous_results: Results from previous subtasks

        Returns:
            Search results (RetrievalResults object)
        """
        if self.graph_store is None or self.retriever is None:
            raise ValueError("Graph store and retriever are required for Search tool")

        # Get search query from tool_params
        search_query = subtask.tool_params.get("query", "")
        if not search_query:
            raise ValueError(f"Search tool requires 'query' parameter in tool_params")

        print(f"  执行检索: {search_query}")

        # Get search generator
        generator = self._get_search_generator()

        # Execute retrieval
        results = generator.retrieve_with_queries(
            input_text=search_query,
            retrieval_mode=self.retrieval_mode,
            entity_top_k=self.entity_top_k,
            timeline_top_k=self.timeline_top_k,
            event_top_k=self.event_top_k,
            question_analysis=query_analysis,
        )

        # Return filtered structured events if available, otherwise merged events
        if hasattr(results, 'filtered_structured_events') and results.filtered_structured_events:
            print(f"  返回过滤后的结构化事件: {len(results.filtered_structured_events)} 条")
            return results.filtered_structured_events
        elif hasattr(results, 'merged_events') and results.merged_events:
            print(f"  返回合并后的事件: {len(results.merged_events)} 个")
            return results.merged_events
        else:
            print(f"  返回原始检索结果")
            return results

    def _execute_python_tool(
        self,
        subtask: SubTask,
        query_analysis: Any,
        previous_results: Dict[str, Any],
    ) -> Any:
        """Execute Python tool

        Args:
            subtask: SubTask object
            query_analysis: QueryParseResult object
            previous_results: Results from previous subtasks

        Returns:
            Python execution result (output string or structured data)
        """
        # Get code description from tool_params
        code_description = subtask.tool_params.get("code_description", "")
        if not code_description:
            raise ValueError(f"Python tool requires 'code_description' parameter in tool_params")

        print(f"  生成并执行 Python 代码")

        # Build context from previous results (depends_on)
        context_parts = []
        if subtask.depends_on:
            for dep_id in subtask.depends_on:
                if dep_id in previous_results:
                    dep_result = previous_results[dep_id]
                    # Format the result as context
                    if isinstance(dep_result, list):
                        context_parts.append(f"Results from {dep_id}: {len(dep_result)} items")
                        # Show first few items
                        if dep_result:
                            context_parts.append(f"Sample: {dep_result[:3]}")
                    elif isinstance(dep_result, str):
                        context_parts.append(f"Results from {dep_id}: {dep_result}")
                    else:
                        context_parts.append(f"Results from {dep_id}: {str(dep_result)[:200]}")

        context = "\n".join(context_parts) if context_parts else ""

        # Get Python LLM
        python_llm = self._get_python_llm()

        # Generate and execute code
        result = python_llm.generate_and_execute(
            user_request=code_description,
            context=context,
            execute=True,
            timeout=30,
        )

        # Extract execution result
        if result.get('execution') and result['execution'].success:
            output = result['execution'].output
            print(f"  Python 执行成功，输出: {output[:100]}...")
            return output
        else:
            error_msg = result['execution'].error if result.get('execution') else "Unknown error"
            raise Exception(f"Python execution failed: {error_msg}")

    def _execute_null_tool(
        self,
        subtask: SubTask,
        query_analysis: Any,
        previous_results: Dict[str, Any],
    ) -> Any:
        """Execute NULL tool (LLM reasoning)

        Args:
            subtask: SubTask object
            query_analysis: QueryParseResult object
            previous_results: Results from previous subtasks

        Returns:
            LLM reasoning result (string)
        """
        print(f"  调用 LLM 进行推理")

        # Build context from previous results
        context_parts = []
        context_parts.append(f"Original Question: {query_analysis.original_question}")
        context_parts.append(f"Question Stem: {query_analysis.question_stem}")
        context_parts.append(f"Time Constraint: {query_analysis.time_constraint.description}")

        # Add previous results
        if subtask.depends_on:
            context_parts.append("\nPrevious Results:")
            for dep_id in subtask.depends_on:
                if dep_id in previous_results:
                    dep_result = previous_results[dep_id]
                    # Format the result
                    if isinstance(dep_result, list):
                        context_parts.append(f"\n{dep_id}: {len(dep_result)} items")
                        # Show details
                        for i, item in enumerate(dep_result[:5], 1):
                            if hasattr(item, '__dict__'):
                                context_parts.append(f"  {i}. {str(item)[:150]}")
                            else:
                                context_parts.append(f"  {i}. {str(item)[:150]}")
                    elif isinstance(dep_result, str):
                        context_parts.append(f"\n{dep_id}: {dep_result}")
                    else:
                        context_parts.append(f"\n{dep_id}: {str(dep_result)[:300]}")

        context = "\n".join(context_parts)

        # Build prompt for LLM
        system_prompt = """You are a helpful assistant that answers temporal questions based on provided context and previous results.

Your task:
1. Analyze the context and previous results
2. Answer the current question directly and concisely
3. Output in JSON format: {"answer": "your answer"}"""

        user_prompt = f"""Context:
{context}

Current Question: {subtask.question}

Please provide your answer based on the context and previous results."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM
        content = self._call_api(messages)
        data = self._parse_json_response(content)

        answer = data.get("answer", "")
        print(f"  LLM 推理结果: {answer[:100]}...")
        return answer

    def _execute_subtask(
        self,
        subtask: SubTask,
        query_analysis: Any,
        previous_results: Dict[str, Any],
    ) -> SubTaskResult:
        """Execute a single subtask

        Args:
            subtask: SubTask object
            query_analysis: QueryParseResult object
            previous_results: Results from previous subtasks (task_id -> result)

        Returns:
            SubTaskResult object
        """
        import time
        start_time = time.time()

        print(f"\n[{subtask.task_id}] {subtask.question}")
        print(f"  工具: {subtask.tool or 'NULL'}")

        try:
            # Execute based on tool type
            if subtask.tool and subtask.tool.lower() == "search":
                result = self._execute_search_tool(subtask, query_analysis, previous_results)
            elif subtask.tool and subtask.tool.lower() == "python":
                result = self._execute_python_tool(subtask, query_analysis, previous_results)
            else:  # NULL tool
                result = self._execute_null_tool(subtask, query_analysis, previous_results)

            execution_time = time.time() - start_time

            return SubTaskResult(
                task_id=subtask.task_id,
                question=subtask.question,
                tool=subtask.tool,
                success=True,
                result=result,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ✗ 执行失败: {e}")

            return SubTaskResult(
                task_id=subtask.task_id,
                question=subtask.question,
                tool=subtask.tool,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
            )

    def execute_task_plan(
        self,
        task_plan: TaskPlan,
    ) -> QueryExecutionResult:
        """Execute a complete task plan

        Args:
            task_plan: TaskPlan object

        Returns:
            QueryExecutionResult object
        """
        import time
        start_time = time.time()

        print("\n" + "=" * 60)
        print("开始执行任务规划")
        print("=" * 60)
        print(f"查询: {task_plan.original_query}")
        print(f"子任务数量: {task_plan.total_tasks}")

        # Store results by task_id
        previous_results = {}
        subtask_results = []

        # Execute subtasks in order
        for subtask in task_plan.subtasks:
            # Check dependencies
            if subtask.depends_on:
                print(f"  依赖: {', '.join(subtask.depends_on)}")
                # Verify all dependencies are available
                for dep_id in subtask.depends_on:
                    if dep_id not in previous_results:
                        raise ValueError(f"Missing dependency: {dep_id} for task {subtask.task_id}")

            # Execute subtask
            result = self._execute_subtask(subtask, task_plan.query_analysis, previous_results)

            # Store result
            subtask_results.append(result)
            if result.success:
                previous_results[subtask.task_id] = result.result
                print(f"  ✓ 执行成功 ({result.execution_time:.2f}s)")
            else:
                # If a subtask fails, stop execution
                print(f"  ✗ 执行失败，停止后续任务")
                break

        # Determine final result (last subtask's result)
        final_result = None
        success = False
        if subtask_results and subtask_results[-1].success:
            final_result = subtask_results[-1].result
            success = True

        total_execution_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("执行完成")
        print("=" * 60)
        print(f"总耗时: {total_execution_time:.2f}s")
        print(f"成功: {success}")
        if success:
            print(f"最终结果: {str(final_result)[:200]}...")

        return QueryExecutionResult(
            original_query=task_plan.original_query,
            task_plan=task_plan,
            subtask_results=subtask_results,
            final_result=final_result,
            success=success,
            total_execution_time=total_execution_time,
            timestamp=datetime.now().isoformat(),
        )
