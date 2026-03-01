"""
Task Planning and Execution Command-Line Interface

Provides interactive and single-command modes for task planning and execution.

Usage:
    # Interactive mode
    python -m timeqa_agent.query_cli

    # Plan tasks for a single query
    python -m timeqa_agent.query_cli plan "Which team did Thierry Audel play for in 2013?"

    # Execute a task plan file
    python -m timeqa_agent.query_cli execute --plan task_plan.json -g graph.json

    # Query directly (plan + execute)
    python -m timeqa_agent.query_cli query "Which team did X play for in 2013?" -g graph.json

    # JSON output
    python -m timeqa_agent.query_cli plan "Who was Anna Karina married to?" --json

    # Save result
    python -m timeqa_agent.query_cli plan "How many goals did he score?" --save --output my_plan.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeqa_agent.query import TaskPlanner, TaskPlan, SubTask, QueryExecutor, QueryExecutionResult
from timeqa_agent.config import load_config


# ============================================================
# Output Functions
# ============================================================

def print_json(data, indent: int = 2):
    """Format and print JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_task_plan(plan: TaskPlan, verbose: bool = False):
    """Print formatted task plan result

    Args:
        plan: TaskPlan object
        verbose: Whether to show detailed information
    """
    print("\n" + "=" * 60)
    print("Task Planning Result")
    print("=" * 60)

    print(f"\nOriginal Query: {plan.original_query}")
    print(f"Total Subtasks: {plan.total_tasks}")

    # Print query analysis summary
    print(f"\nQuery Analysis:")
    print(f"  Question Stem: {plan.query_analysis.question_stem}")
    print(f"  Time Constraint: {plan.query_analysis.time_constraint.constraint_type.value}")
    if plan.query_analysis.time_constraint.description:
        print(f"    - {plan.query_analysis.time_constraint.description}")
    print(f"  Event Type: {plan.query_analysis.event_type.value}")
    print(f"  Answer Type: {plan.query_analysis.answer_type.value}")

    # Print subtasks
    print(f"\n{'=' * 60}")
    print(f"Subtasks")
    print(f"{'=' * 60}")

    for i, task in enumerate(plan.subtasks, 1):
        print(f"\n[{task.task_id}] {task.question}")
        print(f"  Tool: {task.tool or 'NULL'}")

        if task.tool_params:
            print(f"  Parameters:")
            for key, value in task.tool_params.items():
                # Pretty print parameter value
                if isinstance(value, str) and len(value) > 60:
                    print(f"    - {key}: {value[:57]}...")
                else:
                    print(f"    - {key}: {value}")

        if task.depends_on:
            print(f"  Depends On: {', '.join(task.depends_on)}")

        if verbose and task.reasoning:
            print(f"  Reasoning: {task.reasoning}")

    if verbose and plan.timestamp:
        print(f"\nTimestamp: {plan.timestamp}")

    print()


def print_execution_result(result: QueryExecutionResult, verbose: bool = False):
    """Print formatted execution result

    Args:
        result: QueryExecutionResult object
        verbose: Whether to show detailed information
    """
    print("\n" + "=" * 60)
    print("Query Execution Result")
    print("=" * 60)

    print(f"\nOriginal Query: {result.original_query}")
    print(f"Success: {result.success}")
    print(f"Total Execution Time: {result.total_execution_time:.2f}s")

    # Print subtask results
    print(f"\n{'=' * 60}")
    print(f"Subtask Execution Results ({len(result.subtask_results)} tasks)")
    print(f"{'=' * 60}")

    for i, task_result in enumerate(result.subtask_results, 1):
        status = "✓" if task_result.success else "✗"
        print(f"\n{status} [{task_result.task_id}] {task_result.question}")
        print(f"  Tool: {task_result.tool or 'NULL'}")
        print(f"  Status: {'Success' if task_result.success else 'Failed'}")
        print(f"  Execution Time: {task_result.execution_time:.2f}s")

        if task_result.success:
            # Print result preview
            result_preview = str(task_result.result)[:200]
            if isinstance(task_result.result, list):
                print(f"  Result: {len(task_result.result)} items")
                if verbose and task_result.result:
                    for j, item in enumerate(task_result.result[:3], 1):
                        print(f"    {j}. {str(item)[:100]}...")
            else:
                print(f"  Result: {result_preview}...")
        else:
            print(f"  Error: {task_result.error}")

    # Print final result
    print(f"\n{'=' * 60}")
    print("Final Result")
    print(f"{'=' * 60}")

    if result.final_result:
        if isinstance(result.final_result, list):
            print(f"\nResult Type: List ({len(result.final_result)} items)")
            for i, item in enumerate(result.final_result[:10], 1):
                print(f"{i}. {str(item)[:150]}")
            if len(result.final_result) > 10:
                print(f"... and {len(result.final_result) - 10} more items")
        elif isinstance(result.final_result, str):
            print(f"\n{result.final_result}")
        else:
            print(f"\n{str(result.final_result)[:500]}")
    else:
        print("\nNo result (execution failed)")

    print()


def save_task_plan_to_json(
    plan: TaskPlan,
    output_path: Optional[str] = None,
) -> str:
    """Save task plan to JSON file

    Args:
        plan: TaskPlan object
        output_path: Output file path (optional)

    Returns:
        Saved file path
    """
    # Determine output path
    if output_path:
        file_path = Path(output_path)
    else:
        # Default output directory
        output_dir = Path("data/query_plans")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"task_plan_{timestamp}.json"

    # Check if file exists
    existing_data = []

    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Support both single object and list format
                if isinstance(content, list):
                    existing_data = content
                else:
                    existing_data = [content]
        except Exception as e:
            print(f"Warning: Failed to read existing file: {e}, will create new file")
            existing_data = []

    # Append new data
    existing_data.append(plan.to_dict())

    # Save to file
    try:
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        return str(file_path)
    except Exception as e:
        raise Exception(f"Failed to save task plan: {e}")


# ============================================================
# QueryCLI Class
# ============================================================

class QueryCLI:
    """Task Planning and Execution Command-Line Interface"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        verbose: bool = False,
        auto_save: bool = False,
        output_file: str = "query_plan_result.json",
        retrieval_mode: str = "hybrid",
        entity_top_k: int = 5,
        timeline_top_k: int = 10,
        event_top_k: int = 20,
    ):
        """Initialize CLI

        Args:
            config_path: Path to config file
            graph_path: Path to graph file (for execution)
            verbose: Whether to show verbose output
            auto_save: Whether to auto-save results
            output_file: Output file path
            retrieval_mode: Retrieval mode (hybrid/keyword/semantic)
            entity_top_k: Number of entities to retrieve
            timeline_top_k: Number of timelines to retrieve
            event_top_k: Number of events to retrieve
        """
        self.verbose = verbose
        self.auto_save = auto_save
        self.output_file = output_file
        self.retrieval_mode = retrieval_mode
        self.entity_top_k = entity_top_k
        self.timeline_top_k = timeline_top_k
        self.event_top_k = event_top_k

        # Load config
        self.config = load_config(config_path) if config_path else load_config()

        # Create TaskPlanner instance
        try:
            self.planner = TaskPlanner(self.config.query_parser)
        except Exception as e:
            print(f"Warning: Failed to initialize TaskPlanner: {e}")
            print("Please ensure VENUS_API_TOKEN or OPENAI_API_KEY environment variable is set")
            self.planner = None

        # Initialize graph store and retriever (for execution)
        self.graph_store = None
        self.retriever = None
        self.executor = None

        if graph_path:
            self._init_execution_components(graph_path)

    def _init_execution_components(self, graph_path: str):
        """Initialize graph store, retriever, and executor

        Args:
            graph_path: Path to graph file
        """
        if not Path(graph_path).exists():
            print(f"Warning: Graph file not found: {graph_path}")
            print("Execution features will not be available")
            return

        try:
            # Import required modules
            from timeqa_agent.graph_store import TimelineGraphStore
            from timeqa_agent.retrievers import HybridRetriever

            # Load graph store
            print(f"Loading graph: {graph_path}")
            self.graph_store = TimelineGraphStore()
            self.graph_store.load(graph_path)

            # Get graph statistics
            stats = self.graph_store.get_stats()
            print(f"✓ Graph loaded: {stats['nodes']['entities']} entities, "
                  f"{stats['nodes']['events']} events, "
                  f"{stats['nodes']['timelines']} timelines")

            # Create embedding function
            embed_fn = self._create_embed_fn()

            # Create retriever
            self.retriever = HybridRetriever(
                self.graph_store,
                embed_fn=embed_fn,
                config=self.config.retriever,
            )
            print(f"✓ Retriever initialized")

            # Create executor
            self.executor = QueryExecutor(
                config=self.config.query_parser,
                graph_store=self.graph_store,
                retriever=self.retriever,
                retrieval_mode=self.retrieval_mode,
                entity_top_k=self.entity_top_k,
                timeline_top_k=self.timeline_top_k,
                event_top_k=self.event_top_k,
            )
            print(f"✓ Executor initialized")

        except Exception as e:
            print(f"Warning: Failed to initialize execution components: {e}")
            import traceback
            traceback.print_exc()
            self.graph_store = None
            self.retriever = None
            self.executor = None

    def _create_embed_fn(self):
        """Create embedding function"""
        try:
            from timeqa_agent.embeddings import create_embed_fn

            model_type = self.config.retriever.semantic_model_type
            model_name = self.config.retriever.semantic_model_name
            device = self.config.retriever.semantic_model_device

            print(f"Loading embedding model: {model_type} ({model_name})")

            if model_type.lower() == "contriever":
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    device=device,
                    normalize=self.config.retriever.contriever_normalize,
                    batch_size=self.config.retriever.embed_batch_size
                )
            elif model_type.lower() == "dpr":
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    device=device,
                    ctx_encoder_name=self.config.retriever.dpr_ctx_encoder,
                    question_encoder_name=self.config.retriever.dpr_question_encoder,
                    batch_size=self.config.retriever.embed_batch_size
                )
            elif model_type.lower() == "bge-m3":
                if self.config.retriever.bge_m3_model_path:
                    model_name = self.config.retriever.bge_m3_model_path
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    normalize_embeddings=True
                )
            else:
                embed_fn = create_embed_fn(
                    model_type=model_type,
                    model_name=model_name,
                    device=device
                )

            if embed_fn:
                print(f"✓ Embedding model loaded")
            return embed_fn

        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            print("Will use keyword retrieval mode")
            return None

    def cmd_plan(self, query: str) -> TaskPlan:
        """Plan tasks for a query

        Args:
            query: User query

        Returns:
            TaskPlan object
        """
        if self.planner is None:
            raise Exception("TaskPlanner not initialized. Check API token configuration.")

        return self.planner.plan_tasks(query)

    def cmd_execute(self, plan_path: str) -> QueryExecutionResult:
        """Execute a task plan from file

        Args:
            plan_path: Path to task plan JSON file

        Returns:
            QueryExecutionResult object
        """
        if self.executor is None:
            raise Exception("Executor not initialized. Please specify --graph parameter")

        # Load task plan
        with open(plan_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle list format (take the last one)
            if isinstance(data, list):
                data = data[-1]
            task_plan = TaskPlan.from_dict(data)

        print(f"Loaded task plan: {plan_path}")

        # Execute
        return self.executor.execute_task_plan(task_plan)

    def cmd_query(self, query: str) -> QueryExecutionResult:
        """Plan and execute a query (one-step)

        Args:
            query: User query

        Returns:
            QueryExecutionResult object
        """
        if self.planner is None:
            raise Exception("TaskPlanner not initialized. Check API token configuration.")
        if self.executor is None:
            raise Exception("Executor not initialized. Please specify --graph parameter")

        # Step 1: Plan
        print("Step 1: Planning tasks...")
        task_plan = self.planner.plan_tasks(query)
        print(f"✓ Generated {task_plan.total_tasks} subtasks")

        # Step 2: Execute
        print("\nStep 2: Executing tasks...")
        return self.executor.execute_task_plan(task_plan)

    def interactive(self):
        """Interactive mode"""
        print("\n" + "=" * 60)
        print("TimeQA Task Planner & Executor - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  plan <query>         - Plan tasks for a query")
        if self.executor:
            print("  execute <plan_file>  - Execute a task plan file")
            print("  query <query>        - Plan and execute a query (one-step)")
            print("  mode <mode>          - Set retrieval mode (hybrid/keyword/semantic)")
        print("  verbose              - Toggle verbose output")
        print("  json                 - Toggle JSON output mode")
        print("  save                 - Toggle auto-save results")
        print("  help                 - Show help")
        print("  quit/exit            - Exit")
        print("=" * 60)

        if self.executor:
            print(f"\nExecution enabled (retrieval mode: {self.retrieval_mode})")
        else:
            print("\nExecution disabled (no graph file specified)")

        if self.planner is None:
            print("\nWarning: TaskPlanner not initialized. Please set API token first.")
            return

        json_mode = False

        while True:
            try:
                line = input("\n> ").strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print("\nCommands:")
                    print("  plan <query>         - Plan tasks for the query")
                    if self.executor:
                        print("  execute <plan_file>  - Execute a task plan file")
                        print("  query <query>        - Plan and execute a query")
                        print("  mode <mode>          - Set retrieval mode")
                    print("  verbose              - Toggle verbose output")
                    print("  json                 - Toggle JSON output mode")
                    print("  save                 - Toggle auto-save results")
                    print("  quit/exit            - Exit")
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"Verbose output: {'ON' if self.verbose else 'OFF'}")
                elif cmd == "json":
                    json_mode = not json_mode
                    print(f"JSON output: {'ON' if json_mode else 'OFF'}")
                elif cmd == "save":
                    self.auto_save = not self.auto_save
                    print(f"Auto-save: {'ON' if self.auto_save else 'OFF'}")
                    if self.auto_save:
                        print(f"Output file: {self.output_file}")
                elif cmd == "mode" and arg and self.executor:
                    if arg.lower() in ("hybrid", "keyword", "semantic"):
                        self.retrieval_mode = arg.lower()
                        self.executor.retrieval_mode = self.retrieval_mode
                        print(f"Retrieval mode set to: {self.retrieval_mode}")
                    else:
                        print("Usage: mode <hybrid|keyword|semantic>")
                elif cmd == "plan" and arg:
                    try:
                        result = self.cmd_plan(arg)

                        if json_mode:
                            print_json(result.to_dict())
                        else:
                            print_task_plan(result, self.verbose)

                        # Auto-save if enabled
                        if self.auto_save:
                            file_path = save_task_plan_to_json(result, self.output_file)
                            print(f"\nResult saved to: {file_path}")

                    except Exception as e:
                        print(f"Planning failed: {e}")
                        import traceback
                        traceback.print_exc()
                elif cmd == "execute" and arg and self.executor:
                    try:
                        result = self.cmd_execute(arg)

                        if json_mode:
                            print_json(result.to_dict())
                        else:
                            print_execution_result(result, self.verbose)

                    except Exception as e:
                        print(f"Execution failed: {e}")
                        import traceback
                        traceback.print_exc()
                elif cmd == "query" and arg and self.executor:
                    try:
                        result = self.cmd_query(arg)

                        if json_mode:
                            print_json(result.to_dict())
                        else:
                            print_execution_result(result, self.verbose)

                    except Exception as e:
                        print(f"Query failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Default: plan tasks
                    if line:
                        try:
                            result = self.cmd_plan(line)

                            if json_mode:
                                print_json(result.to_dict())
                            else:
                                print_task_plan(result, self.verbose)

                            # Auto-save if enabled
                            if self.auto_save:
                                file_path = save_task_plan_to_json(result, self.output_file)
                                print(f"\nResult saved to: {file_path}")

                        except Exception as e:
                            print(f"Planning failed: {e}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


# ============================================================
# Main Function
# ============================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TimeQA Task Planner & Executor CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m timeqa_agent.query_cli

  # Plan tasks for a question
  python -m timeqa_agent.query_cli plan "Which team did Thierry Audel play for in 2013?"

  # Execute a task plan file
  python -m timeqa_agent.query_cli execute --plan task_plan.json -g data/timeqa/graph/test.json

  # Query directly (plan + execute)
  python -m timeqa_agent.query_cli query "Which team did X play for in 2013?" -g data/timeqa/graph/test.json

  # JSON format output
  python -m timeqa_agent.query_cli plan "Who was Anna Karina married to?" --json

  # Save result to custom file
  python -m timeqa_agent.query_cli plan "How many goals did he score?" --save --output my_plan.json

  # Use custom config file
  python -m timeqa_agent.query_cli -c configs/timeqa_config.json plan "When did he graduate?"

  # Specify retrieval parameters
  python -m timeqa_agent.query_cli query "Who was he married to?" -g graph.json --mode keyword --entity-topk 10
"""
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to config file"
    )

    parser.add_argument(
        "-g", "--graph",
        help="Path to graph file (required for execute/query commands)"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="Command: plan, execute, query, interactive"
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Query question (for plan/query commands)"
    )

    parser.add_argument(
        "--plan",
        help="Path to task plan JSON file (for execute command)"
    )

    parser.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "semantic"],
        default="hybrid",
        help="Retrieval mode (for execute/query commands)"
    )

    parser.add_argument(
        "--entity-topk",
        type=int,
        default=5,
        help="Number of entities to retrieve"
    )

    parser.add_argument(
        "--timeline-topk",
        type=int,
        default=10,
        help="Number of timelines to retrieve"
    )

    parser.add_argument(
        "--event-topk",
        type=int,
        default=20,
        help="Number of events to retrieve"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result to JSON file"
    )

    parser.add_argument(
        "--output",
        default="query_plan_result.json",
        help="Output file path (default: query_plan_result.json)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = QueryCLI(
        config_path=args.config,
        graph_path=args.graph,
        verbose=args.verbose,
        auto_save=args.save,
        output_file=args.output,
        retrieval_mode=args.mode,
        entity_top_k=args.entity_topk,
        timeline_top_k=args.timeline_topk,
        event_top_k=args.event_topk,
    )

    cmd = args.command.lower()

    if cmd == "interactive":
        cli.interactive()
    elif cmd == "plan" and args.query:
        try:
            result = cli.cmd_plan(args.query)

            if args.json:
                print_json(result.to_dict())
            else:
                print_task_plan(result, args.verbose)

            # Save if requested
            if args.save:
                file_path = save_task_plan_to_json(result, args.output)
                print(f"\nResult saved to: {file_path}")

        except Exception as e:
            print(f"Planning failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif cmd == "execute" and args.plan:
        try:
            result = cli.cmd_execute(args.plan)

            if args.json:
                print_json(result.to_dict())
            else:
                print_execution_result(result, args.verbose)

            # Save if requested
            if args.save:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                print(f"\nResult saved to: {output_path}")

        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif cmd == "query" and args.query:
        try:
            result = cli.cmd_query(args.query)

            if args.json:
                print_json(result.to_dict())
            else:
                print_execution_result(result, args.verbose)

            # Save if requested
            if args.save:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                print(f"\nResult saved to: {output_path}")

        except Exception as e:
            print(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
