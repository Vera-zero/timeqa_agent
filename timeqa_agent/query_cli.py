"""
Task Planning Command-Line Interface

Provides interactive and single-command modes for task planning.

Usage:
    # Interactive mode
    python -m timeqa_agent.query_cli

    # Plan tasks for a single query
    python -m timeqa_agent.query_cli plan "Which team did Thierry Audel play for in 2013?"

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

from timeqa_agent.query import TaskPlanner, TaskPlan, SubTask
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
    """Task Planning Command-Line Interface"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
        auto_save: bool = False,
        output_file: str = "query_plan_result.json",
    ):
        """Initialize CLI

        Args:
            config_path: Path to config file
            verbose: Whether to show verbose output
            auto_save: Whether to auto-save results
            output_file: Output file path
        """
        self.verbose = verbose
        self.auto_save = auto_save
        self.output_file = output_file

        # Load config
        self.config = load_config(config_path) if config_path else load_config()

        # Create TaskPlanner instance
        try:
            self.planner = TaskPlanner(self.config.query_parser)
        except Exception as e:
            print(f"Warning: Failed to initialize TaskPlanner: {e}")
            print("Please ensure VENUS_API_TOKEN or OPENAI_API_KEY environment variable is set")
            self.planner = None

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

    def interactive(self):
        """Interactive mode"""
        print("\n" + "=" * 60)
        print("TimeQA Task Planner - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  <query>          - Plan tasks for a query")
        print("  verbose          - Toggle verbose output")
        print("  json             - Toggle JSON output mode")
        print("  save             - Toggle auto-save results")
        print("  help             - Show help")
        print("  quit/exit        - Exit")
        print("=" * 60)

        if self.planner is None:
            print("\nWarning: TaskPlanner not initialized. Please set API token first.")
            return

        json_mode = False

        while True:
            try:
                line = input("\n> ").strip()
                if not line:
                    continue

                cmd = line.lower()

                if cmd in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print("\nCommands:")
                    print("  <query>    - Plan tasks for the query")
                    print("  verbose    - Toggle verbose output")
                    print("  json       - Toggle JSON output mode")
                    print("  save       - Toggle auto-save results")
                    print("  quit/exit  - Exit")
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
                else:
                    # Plan tasks
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
        description="TimeQA Task Planner CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m timeqa_agent.query_cli

  # Plan tasks for a question
  python -m timeqa_agent.query_cli plan "Which team did Thierry Audel play for in 2013?"

  # JSON format output
  python -m timeqa_agent.query_cli plan "Who was Anna Karina married to?" --json

  # Save result to custom file
  python -m timeqa_agent.query_cli plan "How many goals did he score?" --save --output my_plan.json

  # Use custom config file
  python -m timeqa_agent.query_cli -c configs/timeqa_config.json plan "When did he graduate?"
"""
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to config file"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        help="Command: plan, interactive"
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Query question to plan"
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
        verbose=args.verbose,
        auto_save=args.save,
        output_file=args.output,
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
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
