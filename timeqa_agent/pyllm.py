"""
Python Code Generation and Execution Module

使用 LLM 生成 Python 代码并安全执行。

Features:
1. Call LLM to generate Python code based on user requirements
2. Execute generated code in isolated subprocess with timeout
3. Return both generation and execution results
"""

from __future__ import annotations

import os
import sys
import json
import re
import time
import tempfile
import subprocess
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import QueryParserConfig


# ============================================================
# Data Structures
# ============================================================

@dataclass
class CodeGenerationResult:
    """Code generation result"""
    user_request: str                   # Original user request
    generated_code: str                 # Generated Python code
    explanation: Optional[str] = None   # Code explanation
    timestamp: Optional[str] = None     # Generation timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_request": self.user_request,
            "generated_code": self.generated_code,
            "explanation": self.explanation,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeGenerationResult":
        return cls(
            user_request=data.get("user_request", ""),
            generated_code=data.get("generated_code", ""),
            explanation=data.get("explanation"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class CodeExecutionResult:
    """Code execution result"""
    code: str                           # Executed code
    success: bool                       # Execution success
    output: str                         # Standard output
    error: Optional[str] = None         # Error message if any
    execution_time: Optional[float] = None  # Execution time in seconds
    return_code: Optional[int] = None   # Process return code
    timestamp: Optional[str] = None     # Execution timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "return_code": self.return_code,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeExecutionResult":
        return cls(
            code=data.get("code", ""),
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            return_code=data.get("return_code"),
            timestamp=data.get("timestamp"),
        )


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT = """You are an expert Python code generation assistant. Based on user requirements, generate clean, efficient, and executable Python code.

Requirements:
1. Code must be complete and directly executable Python code
2. Use Python standard library; avoid external dependencies unless explicitly requested
3. Include appropriate error handling
4. Use print() function to output results
5. Keep code clean and concise; avoid unnecessary complexity

Output Format (JSON):
{
  "code": "Complete generated Python code",
  "explanation": "Brief explanation of the code functionality"
}

Example:
User Request: Calculate the 10th Fibonacci number
Output:
{
  "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    a, b = 0, 1\\n    for _ in range(2, n + 1):\\n        a, b = b, a + b\\n    return b\\n\\nresult = fibonacci(10)\\nprint(f'The 10th Fibonacci number: {result}')",
  "explanation": "Defines a fibonacci function to calculate the nth term using iteration to avoid recursion overhead"
}
"""

USER_PROMPT = """User Request: {user_request}

{context_section}

Please generate Python code:"""


# ============================================================
# PythonLLM Class
# ============================================================

class PythonLLM:
    """Python code generation and execution using LLM

    Uses QueryParserConfig for API configuration.
    Executes code in isolated subprocess for safety.
    """

    def __init__(
        self,
        config: Optional[QueryParserConfig] = None,
        token: Optional[str] = None,
    ):
        """Initialize PythonLLM

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

    def generate_code(
        self,
        user_request: str,
        context: str = "",
    ) -> CodeGenerationResult:
        """Generate Python code based on user request

        Args:
            user_request: User's code requirement
            context: Additional context (optional)

        Returns:
            CodeGenerationResult object
        """
        # Build context section
        context_section = ""
        if context:
            context_section = f"\nAdditional Context:\n{context}"

        # Build user prompt
        user_prompt = USER_PROMPT.format(
            user_request=user_request,
            context_section=context_section,
        )

        # Call LLM API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = self._call_api(messages)
        data = self._parse_json_response(content)

        # Extract code and explanation
        generated_code = data.get("code", "")
        explanation = data.get("explanation", "")

        return CodeGenerationResult(
            user_request=user_request,
            generated_code=generated_code,
            explanation=explanation,
            timestamp=datetime.now().isoformat(),
        )

    def execute_code(
        self,
        code: str,
        timeout: int = 30,
    ) -> CodeExecutionResult:
        """Execute Python code in isolated subprocess

        Args:
            code: Python code to execute
            timeout: Timeout in seconds (default: 30)

        Returns:
            CodeExecutionResult object
        """
        start_time = time.time()

        # Write code to temporary file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8',
            ) as f:
                f.write(code)
                temp_file = f.name

            # Execute code using subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                # For Windows: CREATE_NO_WINDOW flag
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )

            execution_time = time.time() - start_time

            return CodeExecutionResult(
                code=code,
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time=execution_time,
                return_code=result.returncode,
                timestamp=datetime.now().isoformat(),
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return CodeExecutionResult(
                code=code,
                success=False,
                output="",
                error=f"Execution timeout (exceeded {timeout} seconds)",
                execution_time=execution_time,
                return_code=-1,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return CodeExecutionResult(
                code=code,
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=execution_time,
                return_code=-1,
                timestamp=datetime.now().isoformat(),
            )

        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def generate_and_execute(
        self,
        user_request: str,
        context: str = "",
        execute: bool = True,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Generate code and optionally execute it

        Args:
            user_request: User's code requirement
            context: Additional context (optional)
            execute: Whether to execute generated code (default: True)
            timeout: Execution timeout in seconds (default: 30)

        Returns:
            Dictionary containing generation and execution results
        """
        # Generate code
        generation_result = self.generate_code(user_request, context)

        result = {
            "generation": generation_result,
            "execution": None,
        }

        # Execute code if requested
        if execute and generation_result.generated_code:
            execution_result = self.execute_code(
                generation_result.generated_code,
                timeout=timeout,
            )
            result["execution"] = execution_result

        return result
