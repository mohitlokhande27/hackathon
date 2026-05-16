"""
Multi-Agent Natural Language to Code System
Scaler x IITR Cohort 2 Hackathon

Architecture (LangGraph):
  User Request
      ↓
  Router Agent          ← classifies language (Python/SQL) and task type
      ↓
  Requirements Agent    ← extracts function name, params, expected output
      ↓
  ┌──────────────────────────────┐
  │                              │
  Python Code Agent         SQL Code Agent
  │                              │
  └──────────────┬───────────────┘
                 ↓
        Test Generation Agent   ← generates pytest / assert tests
                 ↓
        Code Executor Agent     ← runs code + tests in sandbox
                 ↓
     [Pass] ──► Response Agent  ← formats final output
     [Fail] ──► Code Refinement Agent ──► Test Gen (retry loop)
"""

import os
import re
import sys
import subprocess
import tempfile
import textwrap
from typing import TypedDict, Optional, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import gradio as gr


# ──────────────────────────────────────────────────────────────────────────────
# 1. State Schema
# ──────────────────────────────────────────────────────────────────────────────

class CodeGenState(TypedDict):
    user_request: str          # original natural-language request
    language: str              # "python" | "sql"
    task_type: str             # "function" | "class" | "script" | "query"
    requirements: str          # structured requirements extracted by Requirements Agent
    generated_code: str        # code produced by Code Gen Agent
    generated_tests: str       # tests produced by Test Gen Agent
    execution_output: str      # stdout/stderr from running the code
    test_output: str           # stdout/stderr from running the tests
    refinement_feedback: str   # feedback for Code Refinement Agent
    final_response: str        # human-readable final answer
    error: str                 # any pipeline error
    iteration: int             # refinement loop counter (max 2)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LLM Setup (Claude via Anthropic)
# ──────────────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0) -> ChatAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=temperature,
        anthropic_api_key=api_key,
    )


def call_llm(system: str, user: str) -> str:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Agent Nodes
# ──────────────────────────────────────────────────────────────────────────────

def router_agent(state: CodeGenState) -> CodeGenState:
    """
    Router Agent: Classifies the user request into language and task type.
    Routes to Python Code Agent or SQL Code Agent.
    """
    print("\n[Router Agent] Classifying request...")

    system = """You are a routing agent for a code generation system.
Analyze the user's request and respond with EXACTLY this JSON (no markdown, no explanation):
{
  "language": "python" or "sql",
  "task_type": "function" or "class" or "script" or "query"
}

Rules:
- If the request mentions SQL, database query, SELECT, INSERT, table → language = "sql"
- Otherwise → language = "python"
- If it asks for a function → task_type = "function"
- If it asks for a class → task_type = "class"
- If it asks for a SQL query → task_type = "query"
- Otherwise → task_type = "script"
"""
    response = call_llm(system, state["user_request"])

    # Parse JSON safely
    lang_match = re.search(r'"language"\s*:\s*"(\w+)"', response)
    type_match = re.search(r'"task_type"\s*:\s*"(\w+)"', response)

    language = lang_match.group(1).lower() if lang_match else "python"
    task_type = type_match.group(1).lower() if type_match else "function"

    print(f"[Router Agent] Language={language}, Task={task_type}")

    return {
        **state,
        "language": language,
        "task_type": task_type,
        "iteration": 0,
    }


def requirements_agent(state: CodeGenState) -> CodeGenState:
    """
    Requirements Agent: Extracts structured requirements from the natural-language request.
    Identifies function/class name, parameters, return type, edge cases, constraints.
    """
    print("\n[Requirements Agent] Extracting structured requirements...")

    system = f"""You are a requirements extraction agent for a {state['language'].upper()} code generation pipeline.
Given a user's natural-language coding request, extract and structure the requirements clearly.

Output a concise structured specification:
- Task: what the code should do
- Name: suggested function/class/variable name
- Inputs: parameters with types (if applicable)
- Output: return type and value description
- Edge cases: empty input, None, negative numbers, etc.
- Constraints: any special rules or algorithmic requirements mentioned
"""
    requirements = call_llm(system, state["user_request"])
    print(f"[Requirements Agent] Done.\n{requirements[:200]}...")

    return {**state, "requirements": requirements}


def python_code_agent(state: CodeGenState) -> CodeGenState:
    """
    Python Code Agent: Generates Python code based on structured requirements.
    Output is clean, executable Python code with no markdown fences.
    """
    print("\n[Python Code Agent] Generating Python code...")

    system = """You are an expert Python developer. Generate clean, working Python code.

Rules:
- Output ONLY the Python code, no markdown code fences (no ```python), no explanations
- Include type hints
- Handle edge cases mentioned in requirements
- Use meaningful variable and function names
- Add a brief docstring
- The code must be directly executable
"""
    user = f"""User Request: {state['user_request']}

Structured Requirements:
{state['requirements']}

Generate the Python {state['task_type']}:"""

    code = call_llm(system, user)
    # Strip markdown fences if LLM added them anyway
    code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE).strip()

    print(f"[Python Code Agent] Generated {len(code.splitlines())} lines of code.")
    return {**state, "generated_code": code}


def sql_code_agent(state: CodeGenState) -> CodeGenState:
    """
    SQL Code Agent: Generates SQL code based on structured requirements.
    Also wraps it with a Python sqlite3 test harness for execution.
    """
    print("\n[SQL Code Agent] Generating SQL code...")

    system = """You are an expert SQL developer. Generate clean, working SQL code.

Rules:
- Output ONLY the SQL query/statement, no markdown fences, no explanations
- Use standard SQL compatible with SQLite
- Use meaningful table and column aliases
- The query must be directly executable in SQLite
"""
    user = f"""User Request: {state['user_request']}

Structured Requirements:
{state['requirements']}

Generate the SQL {state['task_type']}:"""

    sql_code = call_llm(system, user)
    sql_code = re.sub(r"^```(?:sql)?\n?", "", sql_code, flags=re.MULTILINE)
    sql_code = re.sub(r"\n?```$", "", sql_code, flags=re.MULTILINE).strip()

    # Wrap SQL in a Python sqlite3 harness so we can execute it
    python_wrapper = f'''import sqlite3

# SQL Code Agent Output
SQL_QUERY = """{sql_code}"""

def run_sql_demo():
    """Demonstrates the SQL query using an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create sample tables based on query (generic demo setup)
    try:
        # Try to execute the query directly (may need sample data)
        cursor.execute(SQL_QUERY)
        results = cursor.fetchall()
        print("SQL Query executed successfully.")
        print(f"Results: {{results}}")
    except Exception as e:
        print(f"Note: {{e}} - Query may need specific table setup.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_sql_demo()
'''

    print(f"[SQL Code Agent] Generated SQL + Python wrapper.")
    return {**state, "generated_code": python_wrapper}


def test_gen_agent(state: CodeGenState) -> CodeGenState:
    """
    Test Generation Agent: Creates pytest unit tests for the generated code.
    Tests must import and call the generated functions/classes.
    """
    print("\n[Test Generation Agent] Generating unit tests...")

    system = """You are a testing expert. Generate pytest unit tests for the provided code.

Rules:
- Output ONLY Python test code, no markdown fences, no explanations
- Use pytest style (functions starting with test_)
- Use assert statements to verify correctness
- Test normal cases AND edge cases
- Import or copy the function being tested inline (don't assume it's in a separate file)
- Tests must be self-contained and directly runnable with: python -m pytest test_code.py -v
- Include at least 3 meaningful test cases
"""
    user = f"""Original Request: {state['user_request']}

Requirements:
{state['requirements']}

Generated Code to Test:
{state['generated_code']}

Generate comprehensive pytest unit tests:"""

    tests = call_llm(system, user)
    tests = re.sub(r"^```(?:python)?\n?", "", tests, flags=re.MULTILINE)
    tests = re.sub(r"\n?```$", "", tests, flags=re.MULTILINE).strip()

    print(f"[Test Generation Agent] Generated {len(tests.splitlines())} lines of tests.")
    return {**state, "generated_tests": tests}


def code_executor_agent(state: CodeGenState) -> CodeGenState:
    """
    Code Executor Agent: Runs the generated code and tests in an isolated subprocess.
    Captures stdout/stderr and determines pass/fail.
    """
    print("\n[Code Executor Agent] Running code and tests...")

    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "solution.py")
        test_file = os.path.join(tmpdir, "test_solution.py")

        # Write generated code
        with open(code_file, "w") as f:
            f.write(state["generated_code"])

        # Write tests (prepend sys.path so tests can import solution)
        test_preamble = f'import sys\nsys.path.insert(0, "{tmpdir}")\n'
        with open(test_file, "w") as f:
            f.write(test_preamble + state["generated_tests"])

        # Run the code itself
        exec_result = subprocess.run(
            [sys.executable, code_file],
            capture_output=True, text=True, timeout=30, cwd=tmpdir
        )
        execution_output = exec_result.stdout + exec_result.stderr

        # Run the tests with pytest
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=60, cwd=tmpdir
        )
        test_output = test_result.stdout + test_result.stderr

    passed = test_result.returncode == 0
    print(f"[Code Executor Agent] Tests {'PASSED' if passed else 'FAILED'}.")
    print(f"Test output preview:\n{test_output[:300]}")

    # Determine refinement feedback if tests failed
    feedback = ""
    if not passed:
        feedback = f"Tests failed. Here is the error output:\n{test_output}"

    return {
        **state,
        "execution_output": execution_output,
        "test_output": test_output,
        "refinement_feedback": feedback,
        "iteration": state.get("iteration", 0) + 1,
    }


def refinement_agent(state: CodeGenState) -> CodeGenState:
    """
    Code Refinement Agent: Fixes the generated code based on test failure feedback.
    Called only when tests fail (up to 2 retry iterations).
    """
    print(f"\n[Refinement Agent] Iteration {state['iteration']}: Fixing code based on test failures...")

    system = """You are a code debugging expert. Fix the provided Python code based on test failure output.

Rules:
- Output ONLY the corrected Python code, no markdown fences, no explanations
- Preserve the original function/class names and signatures
- Fix the specific errors shown in the test output
- Maintain type hints and docstrings
"""
    user = f"""Original Request: {state['user_request']}

Current Code (needs fixing):
{state['generated_code']}

Test Failure Output:
{state['refinement_feedback']}

Provide the corrected code:"""

    fixed_code = call_llm(system, user)
    fixed_code = re.sub(r"^```(?:python)?\n?", "", fixed_code, flags=re.MULTILINE)
    fixed_code = re.sub(r"\n?```$", "", fixed_code, flags=re.MULTILINE).strip()

    print("[Refinement Agent] Code refined.")
    return {**state, "generated_code": fixed_code}


def response_agent(state: CodeGenState) -> CodeGenState:
    """
    Response Agent: Formats the final human-readable response with code, tests, and results.
    """
    print("\n[Response Agent] Formatting final response...")

    tests_passed = "FAILED" not in state["test_output"] and "error" not in state["test_output"].lower()
    status = "✅ PASSED" if tests_passed else "⚠️ PARTIAL (see test output)"

    response = f"""## Code Generation Result

**Request:** {state['user_request']}
**Language:** {state['language'].upper()} | **Type:** {state['task_type'].title()}

---

### Generated Code
```{state['language']}
{state['generated_code']}
```

---

### Generated Tests
```python
{state['generated_tests']}
```

---

### Test Results: {status}
```
{state['test_output'][:1000]}
```
"""

    return {**state, "final_response": response}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Conditional Edge Functions (Router Logic)
# ──────────────────────────────────────────────────────────────────────────────

def route_to_code_agent(state: CodeGenState) -> Literal["python_code_agent", "sql_code_agent"]:
    """Routes from Requirements Agent to the appropriate Code Agent."""
    if state["language"] == "sql":
        return "sql_code_agent"
    return "python_code_agent"


def route_after_execution(state: CodeGenState) -> Literal["refinement_agent", "response_agent"]:
    """After execution, decide whether to refine or finalize."""
    tests_failed = (
        "FAILED" in state["test_output"]
        or "error" in state["test_output"].lower()
    )
    max_iterations_reached = state.get("iteration", 0) >= 2

    if tests_failed and not max_iterations_reached:
        return "refinement_agent"
    return "response_agent"


def route_after_refinement(state: CodeGenState) -> Literal["test_gen_agent"]:
    """After refinement, always re-run test generation."""
    return "test_gen_agent"


# ──────────────────────────────────────────────────────────────────────────────
# 5. LangGraph Workflow Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """Constructs and compiles the LangGraph state machine."""

    workflow = StateGraph(CodeGenState)

    # ── Add all agent nodes ──
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("requirements_agent", requirements_agent)
    workflow.add_node("python_code_agent", python_code_agent)
    workflow.add_node("sql_code_agent", sql_code_agent)
    workflow.add_node("test_gen_agent", test_gen_agent)
    workflow.add_node("code_executor_agent", code_executor_agent)
    workflow.add_node("refinement_agent", refinement_agent)
    workflow.add_node("response_agent", response_agent)

    # ── Set entry point ──
    workflow.set_entry_point("router_agent")

    # ── Add edges ──
    workflow.add_edge("router_agent", "requirements_agent")

    # Conditional: Python or SQL code agent
    workflow.add_conditional_edges(
        "requirements_agent",
        route_to_code_agent,
        {
            "python_code_agent": "python_code_agent",
            "sql_code_agent": "sql_code_agent",
        },
    )

    # Both code agents feed into test generation
    workflow.add_edge("python_code_agent", "test_gen_agent")
    workflow.add_edge("sql_code_agent", "test_gen_agent")
    workflow.add_edge("test_gen_agent", "code_executor_agent")

    # Conditional: Refine or finalize
    workflow.add_conditional_edges(
        "code_executor_agent",
        route_after_execution,
        {
            "refinement_agent": "refinement_agent",
            "response_agent": "response_agent",
        },
    )

    # After refinement → re-run tests
    workflow.add_edge("refinement_agent", "test_gen_agent")

    # End
    workflow.add_edge("response_agent", END)

    return workflow.compile()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main Pipeline Entry Point
# ──────────────────────────────────────────────────────────────────────────────

graph = build_workflow()


def run_pipeline(user_request: str) -> str:
    """Runs the full multi-agent pipeline for a given coding request."""
    if not user_request.strip():
        return "Please enter a coding request."

    initial_state: CodeGenState = {
        "user_request": user_request,
        "language": "",
        "task_type": "",
        "requirements": "",
        "generated_code": "",
        "generated_tests": "",
        "execution_output": "",
        "test_output": "",
        "refinement_feedback": "",
        "final_response": "",
        "error": "",
        "iteration": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
        return final_state.get("final_response", "No response generated.")
    except Exception as e:
        return f"Pipeline error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Gradio Chat Interface
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_REQUESTS = [
    "Write a Python function that checks if a string is a palindrome",
    "Write a Python function to find the nth Fibonacci number using memoization",
    "Write a Python class for a Stack data structure with push, pop, peek methods",
    "Write a Python function that takes a list and returns only even numbers",
    "Write a SQL query to find the top 3 highest-paid employees from an employees table",
    "Write a Python function to merge two sorted lists into one sorted list",
]


def gradio_handler(message: str, history: list) -> str:
    """Gradio ChatInterface callback."""
    return run_pipeline(message)


def launch_ui():
    """Launches the Gradio chat interface."""
    demo = gr.ChatInterface(
        fn=gradio_handler,
        title="Multi-Agent NL to Code System",
        description=(
            "Router Agent → Requirements Agent → Code Agent (Python/SQL) → "
            "Test Agent → Executor → Response\n\n"
            "Powered by Claude + LangGraph"
        ),
        theme=gr.themes.Soft(),
        examples=EXAMPLE_REQUESTS,
        cache_examples=False,
    )
    demo.launch(debug=True, share=True)


# ──────────────────────────────────────────────────────────────────────────────
# 8. CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent NL to Code System")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--request", type=str, help="Run a single request via CLI")
    args = parser.parse_args()

    if args.ui:
        launch_ui()
    elif args.request:
        result = run_pipeline(args.request)
        print("\n" + "=" * 60)
        print(result)
    else:
        # Default: interactive CLI loop
        print("Multi-Agent NL to Code System (Claude + LangGraph)")
        print("Type 'exit' to quit, '--ui' to launch Gradio interface\n")
        while True:
            user_input = input("Enter coding request: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input == "--ui":
                launch_ui()
                break
            result = run_pipeline(user_input)
            print("\n" + "=" * 60)
            print(result)
            print("=" * 60 + "\n")
