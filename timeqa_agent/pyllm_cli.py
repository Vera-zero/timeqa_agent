"""
Python Code Generation CLI Tool

Command-line interface for generating and executing Python code using LLM.

Features:
- Generate code only (without execution)
- Generate and execute code
- Interactive mode
- Save results to JSON file
"""

import argparse
import json
import sys
import os
from typing import Optional
from datetime import datetime

from .config import load_config
from .pyllm import PythonLLM, CodeGenerationResult, CodeExecutionResult


def print_json(data, indent: int = 2):
    """Print formatted JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def print_generation_result(result: CodeGenerationResult, verbose: bool = False):
    """Print code generation result"""
    print("\n" + "=" * 60)
    print("Code Generation Result")
    print("=" * 60)

    print(f"\nUser Request: {result.user_request}")

    if result.explanation:
        print(f"\nExplanation: {result.explanation}")

    print(f"\nGenerated Code:")
    print("-" * 60)
    print(result.generated_code)
    print("-" * 60)

    if verbose and result.timestamp:
        print(f"\nTimestamp: {result.timestamp}")
    print()


def print_execution_result(result: CodeExecutionResult, verbose: bool = False):
    """Print code execution result"""
    print("\n" + "=" * 60)
    print("Execution Result")
    print("=" * 60)

    status = "✓ Success" if result.success else "✗ Failed"
    print(f"\nStatus: {status}")

    if result.return_code is not None:
        print(f"Return Code: {result.return_code}")

    if result.execution_time is not None:
        print(f"Execution Time: {result.execution_time:.3f}s")

    if result.output:
        print(f"\nOutput:")
        print("-" * 60)
        print(result.output)
        print("-" * 60)

    if result.error:
        print(f"\nError:")
        print("-" * 60)
        print(result.error)
        print("-" * 60)

    if verbose and result.timestamp:
        print(f"\nTimestamp: {result.timestamp}")
    print()


def save_result_to_json(
    result: dict,
    output_path: Optional[str] = None,
) -> str:
    """Save result to JSON file

    Args:
        result: Result dictionary containing generation and execution results
        output_path: Optional output file path

    Returns:
        Path to saved file
    """
    # Create output directory if not exists
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        file_path = output_path
    else:
        # Default output directory
        output_dir = "data/pyllm_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"pyllm_result_{timestamp}.json")

    # Prepare data for JSON serialization
    data = {
        "generation": result["generation"].to_dict() if result.get("generation") else None,
        "execution": result["execution"].to_dict() if result.get("execution") else None,
    }

    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


class PythonLLMCLI:
    """Python LLM command-line interface"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
        auto_save: bool = True,
        output_path: Optional[str] = None,
    ):
        """Initialize CLI

        Args:
            config_path: Path to config file
            verbose: Enable verbose output
            auto_save: Auto-save results to JSON
            output_path: Custom output file path
        """
        self.verbose = verbose
        self.auto_save = auto_save
        self.output_path = output_path

        # Load config
        self.config = load_config(config_path) if config_path else load_config()

        # Create PythonLLM instance
        self.llm = PythonLLM(self.config.query_parser)

    def cmd_generate(
        self,
        request: str,
        context: str = "",
    ) -> CodeGenerationResult:
        """Generate code only (without execution)

        Args:
            request: User's code requirement
            context: Additional context

        Returns:
            CodeGenerationResult
        """
        return self.llm.generate_code(request, context)

    def cmd_execute(
        self,
        request: str,
        context: str = "",
        timeout: int = 30,
    ) -> dict:
        """Generate and execute code

        Args:
            request: User's code requirement
            context: Additional context
            timeout: Execution timeout

        Returns:
            Dictionary containing generation and execution results
        """
        return self.llm.generate_and_execute(
            request,
            context=context,
            execute=True,
            timeout=timeout,
        )

    def interactive(self):
        """Interactive mode"""
        print("\n" + "=" * 60)
        print("TimeQA Python Code Generator - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  <request>        - Generate and execute code")
        print("  generate <req>   - Generate code only (no execution)")
        print("  verbose          - Toggle verbose output")
        print("  json             - Toggle JSON output mode")
        print("  autosave         - Toggle auto-save results")
        print("  help             - Show help")
        print("  quit/exit        - Exit")
        print("=" * 60)

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
                    print("  <request>        - Generate and execute code")
                    print("  generate <req>   - Generate code only")
                    print("  verbose          - Toggle verbose output")
                    print("  json             - Toggle JSON output mode")
                    print("  autosave         - Toggle auto-save results")
                    print("  quit/exit        - Exit")
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"Verbose output: {'ON' if self.verbose else 'OFF'}")
                elif cmd == "json":
                    json_mode = not json_mode
                    print(f"JSON output: {'ON' if json_mode else 'OFF'}")
                elif cmd == "autosave":
                    self.auto_save = not self.auto_save
                    print(f"Auto-save: {'ON' if self.auto_save else 'OFF'}")
                elif cmd.startswith("generate "):
                    # Generate only
                    request = line[9:].strip()
                    if not request:
                        print("Error: Please provide a request")
                        continue

                    try:
                        result = self.cmd_generate(request)

                        if json_mode:
                            print_json(result.to_dict())
                        else:
                            print_generation_result(result, self.verbose)

                        # Save to file if auto-save is enabled
                        if self.auto_save:
                            save_result = {"generation": result, "execution": None}
                            file_path = save_result_to_json(save_result, self.output_path)
                            print(f"Result saved to: {file_path}")

                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    # Generate and execute
                    try:
                        result = self.cmd_execute(line)

                        if json_mode:
                            data = {
                                "generation": result["generation"].to_dict() if result.get("generation") else None,
                                "execution": result["execution"].to_dict() if result.get("execution") else None,
                            }
                            print_json(data)
                        else:
                            if result.get("generation"):
                                print_generation_result(result["generation"], self.verbose)
                            if result.get("execution"):
                                print_execution_result(result["execution"], self.verbose)

                        # Save to file if auto-save is enabled
                        if self.auto_save:
                            file_path = save_result_to_json(result, self.output_path)
                            print(f"Result saved to: {file_path}")

                    except Exception as e:
                        print(f"Error: {e}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(
        description="TimeQA Python Code Generation CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m timeqa_agent.pyllm_cli

  # Generate and execute code
  python -m timeqa_agent.pyllm_cli execute "Calculate the sum of 1 to 100"

  # Generate code only (no execution)
  python -m timeqa_agent.pyllm_cli generate "Read CSV file and count rows"

  # Save result to specific file
  python -m timeqa_agent.pyllm_cli execute "Calculate factorial of 10" --save-output result.json

  # JSON format output
  python -m timeqa_agent.pyllm_cli execute "Print current date" --json

  # Use custom config file
  python -m timeqa_agent.pyllm_cli -c configs/timeqa_config.json execute "List files"
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
        help="Command: generate, execute, interactive"
    )

    parser.add_argument(
        "request",
        nargs="?",
        help="Code generation request"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    parser.add_argument(
        "--save-output",
        help="Save result to specified file"
    )

    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Generate code only, do not execute"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not auto-save results"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Execution timeout in seconds (default: 30)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    cli = PythonLLMCLI(
        config_path=args.config,
        verbose=args.verbose,
        auto_save=not args.no_save,
        output_path=args.save_output,
    )

    cmd = args.command.lower()

    if cmd == "interactive":
        cli.interactive()
    elif cmd == "generate" and args.request:
        try:
            result = cli.cmd_generate(args.request)

            if args.json:
                print_json(result.to_dict())
            else:
                print_generation_result(result, args.verbose)

            # Save result
            if not args.no_save:
                save_result = {"generation": result, "execution": None}
                file_path = save_result_to_json(save_result, args.save_output)
                print(f"\nResult saved to: {file_path}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif cmd == "execute" and args.request:
        try:
            if args.no_execute:
                # Generate only
                result_obj = cli.cmd_generate(args.request)
                result = {"generation": result_obj, "execution": None}
            else:
                # Generate and execute
                result = cli.cmd_execute(args.request, timeout=args.timeout)

            if args.json:
                data = {
                    "generation": result["generation"].to_dict() if result.get("generation") else None,
                    "execution": result["execution"].to_dict() if result.get("execution") else None,
                }
                print_json(data)
            else:
                if result.get("generation"):
                    print_generation_result(result["generation"], args.verbose)
                if result.get("execution"):
                    print_execution_result(result["execution"], args.verbose)

            # Save result
            if not args.no_save:
                file_path = save_result_to_json(result, args.save_output)
                print(f"\nResult saved to: {file_path}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
