"""FastMCP server implementation for the Codex MCP project."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from pydantic import BeforeValidator, Field
import shutil

mcp = FastMCP("Codex MCP Server-from guda.studio")


def _empty_str_to_none(value: str | None) -> str | None:
    """Convert empty strings to None for optional UUID parameters."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def run_shell_command(
    cmd: list[str],
    timeout: Optional[float] = None,
) -> Tuple[subprocess.Popen[str], Generator[str, None, None], List[bool]]:
    """Execute a command and stream its output line-by-line with optional timeout.

    Args:
        cmd: Command and arguments as a list (e.g., ["codex", "exec", "prompt"])
        timeout: Optional timeout in seconds. None means no timeout.

    Returns:
        Tuple of (process, output_generator, timeout_flag)
        - process: The subprocess.Popen instance
        - output_generator: Generator yielding output lines
        - timeout_flag: Mutable list [bool] indicating if timeout occurred.
                       Check timeout_flag[0] after consuming the generator.

    Yields:
        Output lines from the command
    """
    # Copy the command list to avoid modifying the caller's list
    popen_cmd = list(cmd)
    codex_path = shutil.which('codex') or cmd[0]
    popen_cmd[0] = codex_path

    process = subprocess.Popen(
        popen_cmd,
        shell=False,  # Safer: no shell injection
        stdin=subprocess.PIPE,  # Prevent process from waiting for input
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
    )

    output_queue: queue.Queue[str] = queue.Queue()
    # Use list to allow mutation in nested function (mutable container pattern)
    timeout_flag = [False]

    def read_output() -> None:
        """Read process output in a separate thread."""
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line.strip())
            process.stdout.close()

    thread = threading.Thread(target=read_output)
    thread.daemon = True
    thread.start()

    start_time = time.monotonic()

    def line_iterator() -> Generator[str, None, None]:
        """Generator that yields output lines with timeout protection."""
        nonlocal start_time

        # Yield lines while process is running
        while process.poll() is None:
            # Check timeout
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                timeout_flag[0] = True
                # Graceful termination: SIGTERM first
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still alive after 2 seconds
                    process.kill()
                    process.wait()
                break

            try:
                yield output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

        # Process has exited, wait for read thread to finish to avoid losing tail output
        thread.join(timeout=1)

        # Drain remaining output from queue
        while not output_queue.empty():
            try:
                yield output_queue.get_nowait()
            except queue.Empty:
                break

    # Return the mutable list so caller can check timeout_flag[0] after consuming iterator
    return process, line_iterator(), timeout_flag


@mcp.tool(
    name="codex",
    description="""
    Executes a non-interactive Codex session via CLI to perform AI-assisted coding tasks in a secure workspace.
    This tool wraps the `codex exec` command, enabling model-driven code generation, debugging, or automation based on natural language prompts.
    It supports resuming ongoing sessions for continuity and enforces sandbox policies to prevent unsafe operations. Ideal for integrating Codex into MCP servers for agentic workflows, such as code reviews or repo modifications.

    **Key Features:**
        - **Prompt-Driven Execution:** Send task instructions to Codex for step-by-step code handling.
        - **Workspace Isolation:** Operate within a specified directory, with optional Git repo skipping.
        - **Security Controls:** Three sandbox levels balance functionality and safety.
        - **Session Persistence:** Resume prior conversations via `SESSION_ID` for iterative tasks.

    **Edge Cases & Best Practices:**
        - Ensure `cd` exists and is accessible; tool fails silently on invalid paths.
        - For most repos, prefer "read-only" to avoid accidental changes.
        - If needed, set `return_all_messages` to `True` to parse "all_messages" for detailed tracing (e.g., reasoning, tool calls, etc.).
    """,
    meta={"version": "0.0.0", "author": "guda.studio"},
)
async def codex(
    PROMPT: Annotated[str, "Instruction for the task to send to codex."],
    cd: Annotated[Path, "Set the workspace root for codex before executing the task."],
    sandbox: Annotated[
        Literal["read-only", "workspace-write", "danger-full-access"],
        Field(
            description="Sandbox policy for model-generated commands. Defaults to `read-only`."
        ),
    ] = "read-only",
    SESSION_ID: Annotated[
        Optional[uuid.UUID],
        BeforeValidator(_empty_str_to_none),
        "Resume the specified session of the codex. Defaults to `None`, start a new session.",
    ] = None,
    skip_git_repo_check: Annotated[
        bool,
        "Allow codex running outside a Git repository (useful for one-off directories).",
    ] = False,
    return_all_messages: Annotated[
        bool,
        "Return all messages (e.g. reasoning, tool calls, etc.) from the codex session. Set to `False` by default, only the agent's final reply message is returned.",
    ] = False,
    image: Annotated[
        Optional[List[Path]],
        Field(
            description="Attach one or more image files to the initial prompt. Separate multiple paths with commas or repeat the flag.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        Field(
            description="The model to use for the codex session. Default user configuration is applied; this parameter remains inactive unless explicitly specified by the user.",
        ),
    ] = None,
    yolo: Annotated[
        Optional[bool],
        Field(
            description="Run every command without approvals or sandboxing. Only use when `sandbox` couldn't be applied.",
        ),
    ] = False,
    profile: Annotated[
        Optional[str],
        "Configuration profile name to load from `~/.codex/config.toml`. Default user configuration is applied; this parameter remains inactive unless explicitly specified by the user.",
    ] = None,
    timeout: Annotated[
        Optional[float],
        Field(
            description="Command execution timeout in seconds. Pass None to disable timeout. Defaults to 600 seconds (10 minutes)."
        ),
    ] = 600.0,
) -> Dict[str, Any]:
    """Execute a Codex CLI session and return the results."""
    # Build command as list to avoid injection
    cmd = ["codex", "exec", "--sandbox", sandbox, "--cd", str(cd), "--json"]

    # Fix: Handle image paths correctly by adding each as separate --image argument
    # This avoids TypeError with Path objects and handles paths with commas/spaces
    if image is not None:
        for img_path in image:
            cmd.extend(["--image", str(img_path)])

    if model is not None:
        cmd.extend(["--model", model])

    if profile is not None:
        cmd.extend(["--profile", profile])

    if yolo:
        cmd.append("--yolo")

    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    if SESSION_ID is not None:
        cmd.extend(["resume", str(SESSION_ID)])

    # No escaping needed: shell=False means arguments are passed directly
    # Removed harmful windows_escape() that was corrupting prompts
    cmd += ['--', PROMPT]

    all_messages: list[Dict[str, Any]] = []
    agent_messages = ""
    success = True
    err_message = ""
    thread_id: Optional[str] = None
    last_output_line: Optional[str] = None

    # Execute command with timeout protection
    process, output_iter, timeout_flag = run_shell_command(cmd, timeout=timeout)

    # Process output lines
    for line in output_iter:
        last_output_line = line  # Track last line for error reporting
        try:
            line_dict = json.loads(line.strip())
            all_messages.append(line_dict)
            item = line_dict.get("item", {})
            item_type = item.get("type")
            if item_type == "agent_message":
                agent_messages = agent_messages + item.get("text", "")
            if line_dict.get("thread_id") is not None:
                thread_id = line_dict.get("thread_id")
        except json.JSONDecodeError as error:
            # Include problematic line in error message
            err_message = f"JSON decode error: {line}"
            success = False
            break
        except Exception as error:
            err_message = f"Unexpected error: {error}. Line: {line!r}"
            success = False
            break

    # Check timeout flag AFTER consuming the iterator (fix: was checking before iteration)
    timed_out = timeout_flag[0]
    if timed_out:
        success = False
        # Priority: timeout message takes precedence over other errors
        err_message = f"Codex CLI execution timed out after {timeout} seconds"

    # Critical: Check subprocess exit code (P0 fix)
    returncode = process.poll()
    if returncode is None:
        returncode = process.wait()

    # Non-zero exit code indicates failure
    if success and returncode != 0:
        success = False
        err_message = f"Codex CLI exited with non-zero code: {returncode}"

    # Validate required fields only if process succeeded
    if success:
        if thread_id is None:
            success = False
            err_message = "Failed to get `SESSION_ID` from the codex session."

        if len(agent_messages) == 0:
            success = False
            err_message = "Failed to get `agent_messages` from the codex session. Try setting `return_all_messages` to `True` for detailed information."

    # Build unified response structure (success and failure have same shape for easier consumption)
    if success:
        result: Dict[str, Any] = {
            "success": True,
            "SESSION_ID": thread_id,
            "agent_messages": agent_messages,
        }
        if return_all_messages:
            result["all_messages"] = all_messages
    else:
        # Enhanced error response with diagnostic information (P1 improvement)
        result = {
            "success": False,
            "error": err_message,
            "returncode": returncode,
            "timeout": timed_out,
            # Always include all_messages for debugging (even if empty list)
            "all_messages": all_messages,
        }
        # Include partial SESSION_ID if available (helps with debugging)
        if thread_id:
            result["SESSION_ID"] = thread_id
        # Include last output line for troubleshooting
        if last_output_line:
            result["last_output_line"] = last_output_line

    return result


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
