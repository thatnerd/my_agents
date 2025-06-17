#!/usr/bin/env python
"""My Python code companion for personal use.

Created by following the logical pathway in https://ampcode.com/how-to-build-an-agent
(but in Python)

Version: 0.1.10

Usage:
    ./my_python_code_companion.py [options]

Options:
    -h --help              Show this screen.
    -v --verbose           Show available tools on startup.
    --model=<model>        Claude model to use [default: claude-3-haiku-20240307].
    --system=<file>        System prompt file in ../prompt/ directory.
    --max-tokens=<tokens>  Maximum tokens in response [default: 4096].

Commands:
    /exit, /quit           Exit the program.
    /reset                 Reset conversation context and start fresh.
    Enter on empty line    Submit multiline prompt.
    Ctrl-D twice           Exit the program.
"""

import os
import sys
import json
import subprocess
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Literal, Callable, Any
from functools import wraps

import anthropic
from docopt import docopt

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import TerminalFormatter
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes optimized for dark backgrounds."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # User/system colors
    USER = '\033[96m'      # Bright cyan
    CLAUDE = '\033[92m'    # Bright green
    TOOL_INFO = '\033[93m' # Bright yellow
    
    # Code highlighting colors
    CODE_HEADER = '\033[95m'  # Bright magenta
    INLINE_CODE = '\033[91m'  # Bright red


def load_api_key() -> str:
    """Load Anthropic API key from ~/.keys/anthropic_api_key."""
    key_path = Path.home() / ".keys" / "anthropic_api_key"
    return key_path.read_text().strip()


def load_system_prompt(filename: Optional[str]) -> Optional[str]:
    """Load system prompt from ../prompt/ directory."""
    if not filename:
        return None
    
    prompt_path = Path(__file__).parent / ".." / "prompt" / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    
    content = prompt_path.read_text().strip()
    return content if content else None


def tool(name: str, description: str) -> Callable:
    """Decorator to register a method as a tool for Claude.
    
    Args:
        name: The name of the tool as Claude will see it.
        description: Description of what the tool does.
        
    Returns:
        Decorator function that marks the method as a tool.
    """
    def decorator(func: Callable) -> Callable:
        func._tool_name = name
        func._tool_description = description
        func._is_tool = True
        return func
    return decorator


class Conversation:
    """Manages a conversation with Claude, including context and storage."""
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 4096,
        storage_mode: Literal["full", "prompts", "none"] = "full"
    ) -> None:
        """Initialize a new conversation.
        
        Args:
            client: Anthropic client instance.
            system_prompt: Optional system prompt for the conversation.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.
            storage_mode: How to store the conversation:
                - "full": Store complete conversation
                - "prompts": Store only user inputs + system prompt
                - "none": Don't store anything
        """
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.storage_mode = storage_mode
        self.messages: List[Dict[str, str]] = []
        self.transcript: List[Dict[str, str]] = []
        self.working_directory: Path = Path.cwd()
        self.session_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verbose: bool = False
        
        # Discover and register tools
        self.tools = self._discover_tools()
    
    def _discover_tools(self) -> List[Dict[str, Any]]:
        """Discover all methods decorated with @tool and build tool definitions.
        
        Returns:
            List of tool definitions for Claude API.
        """
        tools = []
        
        for method_name in dir(self):
            method = getattr(self, method_name)
            if hasattr(method, '_is_tool'):
                tool_def = self._build_tool_schema(method)
                tools.append(tool_def)
        
        return tools
    
    def _build_tool_schema(self, method: Callable) -> Dict[str, Any]:
        """Build Claude API tool schema from decorated method.
        
        Args:
            method: The decorated method to build schema for.
            
        Returns:
            Tool definition dictionary for Claude API.
        """
        import inspect
        
        # Get method signature
        sig = inspect.signature(method)
        parameters = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Extract type information
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int:
                    param_type = "integer"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            parameters[param_name] = {"type": param_type}
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "name": method._tool_name,
            "description": method._tool_description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
    
    def _validate_path(self, path: str) -> Path:
        """Validate that a path is within the working directory tree.
        
        Args:
            path: The path to validate (relative or absolute).
            
        Returns:
            Resolved Path object if valid.
            
        Raises:
            ValueError: If path escapes the working directory.
        """
        # Handle relative paths
        if os.path.isabs(path):
            target_path = Path(path)
        else:
            target_path = self.working_directory / path
        
        # Resolve to absolute path
        target_path = target_path.resolve()
        
        # Check if path is within working directory
        try:
            target_path.relative_to(self.working_directory.resolve())
        except ValueError:
            raise ValueError(f"Path '{path}' is outside the allowed directory")
        
        return target_path
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name with given input.
        
        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input parameters for the tool.
            
        Returns:
            Tool execution result as string.
        """
        # Find the method for this tool
        for method_name in dir(self):
            method = getattr(self, method_name)
            if hasattr(method, '_is_tool') and method._tool_name == tool_name:
                try:
                    # Call the method with unpacked arguments
                    return method(**tool_input)
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"
        
        return f"Unknown tool: {tool_name}"
    
    def _format_response(self, response: str) -> str:
        """Format Claude's response with syntax highlighting for code blocks and inline code.
        
        Args:
            response: Raw response text from Claude.
            
        Returns:
            Formatted response with syntax highlighting.
        """
        if not PYGMENTS_AVAILABLE:
            return response
        
        # First handle inline code (single backticks, length < 50)
        def highlight_inline_code(match):
            code = match.group(1)
            if len(code) < 50:
                return f"{Colors.INLINE_CODE}`{code}`{Colors.RESET}"
            else:
                return match.group(0)  # Leave long inline code unchanged
        
        # Handle inline code first
        response = re.sub(r'`([^`\n]+)`', highlight_inline_code, response)
        
        # Pattern to match code blocks with optional language specification
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        
        def highlight_code_block(match):
            language = match.group(1) or 'text'
            code = match.group(2)
            
            try:
                if language and language != 'text':
                    lexer = get_lexer_by_name(language, stripall=True)
                else:
                    lexer = guess_lexer(code)
                    language = lexer.name.lower()
                
                # Use a custom formatter for dark backgrounds
                formatter = TerminalFormatter(
                    style='monokai',  # Dark background friendly style
                    bg='dark'
                )
                highlighted = highlight(code, lexer, formatter)
                
                # Add header and remove trailing whitespace
                header = f"{Colors.CODE_HEADER}# <code_block> ({language}){Colors.RESET}\n"
                return f"{header}{highlighted.rstrip()}"
                
            except ClassNotFound:
                # If we can't find a lexer, return with basic formatting
                header = f"{Colors.CODE_HEADER}# <code_block> ({language}){Colors.RESET}\n"
                return f"{header}{code}"
        
        return re.sub(code_block_pattern, highlight_code_block, response, flags=re.DOTALL)
    
    @tool("list_directory", "List contents of a directory. Use '.' for current directory.")
    def list_directory(self, path: str = ".") -> str:
        """List contents of the specified directory.
        
        Args:
            path: Path to the directory to list (relative or absolute).
        
        Returns:
            String representation of directory contents.
        """
        try:
            target_path = self._validate_path(path)
            
            if not target_path.exists():
                return f"Directory does not exist: {path}"
            
            if not target_path.is_dir():
                return f"Path is not a directory: {path}"
            
            # Try to use ls for detailed output
            try:
                result = subprocess.run(
                    ["ls", "-la"],
                    cwd=target_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return f"Directory listing for {target_path}:\n{result.stdout}"
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback for Windows or systems without ls
                files = list(target_path.iterdir())
                file_list = [f"Directory listing for {target_path}:"]
                
                for file_path in sorted(files):
                    if file_path.is_dir():
                        file_list.append(f"d {file_path.name}/")
                    else:
                        file_list.append(f"f {file_path.name}")
                        
                return "\n".join(file_list)
                
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Error listing directory {path}: {str(e)}"
    
    @tool("read_file", "Read the contents of a text file.")
    def read_file(self, filepath: str) -> str:
        """Read and return the contents of a text file.
        
        Args:
            filepath: Path to the file to read (relative or absolute).
            
        Returns:
            File contents as string.
        """
        try:
            target_path = self._validate_path(filepath)
            
            if not target_path.exists():
                return f"File does not exist: {filepath}"
            
            if not target_path.is_file():
                return f"Path is not a file: {filepath}"
            
            # Read file contents
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"Contents of {target_path}:\n{content}"
            
        except ValueError as e:
            return str(e)
        except UnicodeDecodeError:
            return f"Cannot read file {filepath}: appears to be binary or uses unsupported encoding"
        except PermissionError:
            return f"Permission denied reading file: {filepath}"
        except Exception as e:
            return f"Error reading file {filepath}: {str(e)}"
    
    @tool("write_file", "Write content to a text file, creating it if it doesn't exist.")
    def write_file(self, filepath: str, content: str) -> str:
        """Write content to a text file.
        
        Args:
            filepath: Path to the file to write (relative or absolute).
            content: Content to write to the file.
            
        Returns:
            Success or error message.
        """
        try:
            target_path = self._validate_path(filepath)
            
            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file contents
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {target_path}"
            
        except ValueError as e:
            return str(e)
        except PermissionError:
            return f"Permission denied writing to file: {filepath}"
        except Exception as e:
            return f"Error writing file {filepath}: {str(e)}"
    
    @tool("edit_file", "Edit a file by replacing old text with new text.")
    def edit_file(self, filepath: str, old_text: str, new_text: str) -> str:
        """Edit a file by replacing old text with new text.
        
        Args:
            filepath: Path to the file to edit (relative or absolute).
            old_text: Text to find and replace.
            new_text: Text to replace with.
            
        Returns:
            Success or error message with replacement count.
        """
        try:
            target_path = self._validate_path(filepath)
            
            if not target_path.exists():
                return f"File does not exist: {filepath}"
            
            if not target_path.is_file():
                return f"Path is not a file: {filepath}"
            
            # Read current content
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if old_text exists
            if old_text not in content:
                return f"Text to replace not found in {filepath}"
            
            # Replace text
            new_content = content.replace(old_text, new_text)
            replacement_count = content.count(old_text)
            
            # Write back
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return f"Successfully replaced {replacement_count} occurrence(s) in {target_path}"
            
        except ValueError as e:
            return str(e)
        except UnicodeDecodeError:
            return f"Cannot edit file {filepath}: appears to be binary or uses unsupported encoding"
        except PermissionError:
            return f"Permission denied editing file: {filepath}"
        except Exception as e:
            return f"Error editing file {filepath}: {str(e)}"
    
    @tool("copy_file", "Copy a file from source to destination.")
    def copy_file(self, source: str, destination: str) -> str:
        """Copy a file from source to destination.
        
        Args:
            source: Path to the source file (relative or absolute).
            destination: Path to the destination (relative or absolute).
            
        Returns:
            Success or error message.
        """
        try:
            source_path = self._validate_path(source)
            dest_path = self._validate_path(destination)
            
            if not source_path.exists():
                return f"Source file does not exist: {source}"
            
            if not source_path.is_file():
                return f"Source path is not a file: {source}"
            
            # Create destination parent directories if they don't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
            
            return f"Successfully copied {source_path} to {dest_path}"
            
        except ValueError as e:
            return str(e)
        except PermissionError:
            return f"Permission denied copying from {source} to {destination}"
        except Exception as e:
            return f"Error copying file from {source} to {destination}: {str(e)}"
    
    @tool("create_directory", "Create a new directory (and parent directories if needed).")
    def create_directory(self, dirpath: str) -> str:
        """Create a new directory.
        
        Args:
            dirpath: Path to the directory to create (relative or absolute).
            
        Returns:
            Success or error message.
        """
        try:
            target_path = self._validate_path(dirpath)
            
            if target_path.exists():
                if target_path.is_dir():
                    return f"Directory already exists: {dirpath}"
                else:
                    return f"Path exists but is not a directory: {dirpath}"
            
            # Create directory and parents
            target_path.mkdir(parents=True, exist_ok=True)
            
            return f"Successfully created directory: {target_path}"
            
        except ValueError as e:
            return str(e)
        except PermissionError:
            return f"Permission denied creating directory: {dirpath}"
        except Exception as e:
            return f"Error creating directory {dirpath}: {str(e)}"
    
    @tool("delete_file", "Delete a file (use with caution!).")
    def delete_file(self, filepath: str) -> str:
        """Delete a file.
        
        Args:
            filepath: Path to the file to delete (relative or absolute).
            
        Returns:
            Success or error message.
        """
        try:
            target_path = self._validate_path(filepath)
            
            if not target_path.exists():
                return f"File does not exist: {filepath}"
            
            if not target_path.is_file():
                return f"Path is not a file: {filepath}"
            
            # Delete the file
            target_path.unlink()
            
            return f"Successfully deleted file: {target_path}"
            
        except ValueError as e:
            return str(e)
        except PermissionError:
            return f"Permission denied deleting file: {filepath}"
        except Exception as e:
            return f"Error deleting file {filepath}: {str(e)}"
    
    def send_message(self, user_input: str) -> str:
        """Send a message to Claude and get the response.
        
        Args:
            user_input: The user's message.
            
        Returns:
            Claude's response text.
            
        Raises:
            anthropic.APITimeoutError: If the request times out.
        """
        # Add user message to both histories
        user_message = {"role": "user", "content": user_input}
        self.messages.append(user_message)
        self.transcript.append(user_message)
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "tools": self.tools
        }
        
        if self.system_prompt:
            request_params["system"] = self.system_prompt
        
        # Send request to Claude
        response = self.client.messages.create(**request_params)
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            return self._handle_tool_use(response)
        else:
            # Regular response
            assistant_message = response.content[0].text
            
            # Add to both histories
            assistant_response = {"role": "assistant", "content": assistant_message}
            self.messages.append(assistant_response)
            self.transcript.append(assistant_response)
            
            return assistant_message
    
    def _handle_tool_use(self, response) -> str:
        """Handle tool use requests from Claude.
        
        Args:
            response: The Claude response containing tool use requests.
            
        Returns:
            Claude's final response after tool execution.
        """
        # Add Claude's tool request to transcript
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                print(f"{Colors.TOOL_INFO}Claude is running {tool_name}()...{Colors.RESET}")
                
                # Add to transcript
                self.transcript.append({
                    "role": "tool_call", 
                    "content": f"Claude is running {tool_name}()"
                })
                
                # Execute the tool
                tool_input = getattr(content_block, 'input', {})
                tool_result = self._execute_tool(tool_name, tool_input)
                
                # Add tool result to transcript
                self.transcript.append({
                    "role": "tool_result",
                    "content": f"Result: {tool_result}"
                })
                
                # Add Claude's tool request to API messages
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool result for API
                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        }
                    ]
                })
        
        # Get Claude's final response with tool results
        final_request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "tools": self.tools
        }
        
        if self.system_prompt:
            final_request_params["system"] = self.system_prompt
        
        final_response = self.client.messages.create(**final_request_params)
        
        # Handle the final response which might contain text, tool use, or both
        if final_response.stop_reason == "tool_use":
            # If Claude wants to use more tools, handle recursively
            return self._handle_tool_use(final_response)
        else:
            # Extract text content from the response
            final_message = ""
            for content_block in final_response.content:
                if hasattr(content_block, 'text'):
                    final_message += content_block.text
            
            # Add final response to both histories
            final_assistant_response = {"role": "assistant", "content": final_message}
            self.messages.append(final_assistant_response)
            self.transcript.append(final_assistant_response)
            
            return final_message
    
    def reset_conversation(self) -> None:
        """Reset the conversation context and start fresh."""
        # Save current conversation if it has content
        if self.messages:
            self.close()
        
        # Clear conversation state
        self.messages.clear()
        self.transcript.clear()
        
        # Create new session timestamp
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log the reset in transcript
        self.transcript.append({
            "role": "system",
            "content": "Conversation context reset"
        })
        
        print(f"{Colors.TOOL_INFO}Conversation context reset. Starting fresh session.{Colors.RESET}\n")
    
    def _get_input(self) -> Optional[str]:
        """Get user input, supporting multiline like Python REPL.
        
        Users can enter multiple lines, and submit by pressing Enter on empty line.
        
        Returns:
            The complete user input, or None if user wants to exit.
        """
        lines = []
        print(f"{Colors.USER}You:{Colors.RESET} ", end="", flush=True)
        
        try:
            while True:
                line = input()
                
                # Check for exit commands on first line
                if not lines:
                    stripped = line.strip().lower()
                    if stripped in ['/exit', '/quit']:
                        return None
                    elif stripped == '/reset':
                        return '/reset'
                
                # If line is empty and we have content, submit
                if not line.strip() and lines:
                    break
                
                # If line is not empty, add it and continue
                if line.strip():
                    lines.append(line)
                    print("...  ", end="", flush=True)  # Continuation prompt
                elif not lines:
                    # Empty line with no content - show prompt again
                    print(f"{Colors.USER}You:{Colors.RESET} ", end="", flush=True)
                    
        except (EOFError, KeyboardInterrupt):
            if lines:
                # If we have partial input, submit it
                return "\n".join(lines)
            return None
        
        return "\n".join(lines) if lines else None
    
    def interact(self) -> None:
        """Interactive chat loop with the user."""
        print(f"Claude Chat CLI (Model: {self.model})")
        if self.system_prompt:
            print("System prompt loaded")
        print(f"Working directory: {self.working_directory}")
        
        if self.verbose:
            print(f"Available tools: {', '.join(tool['name'] for tool in self.tools)}")
        
        print("Enter messages. Press Enter twice (empty line) to submit.")
        print("Use /exit or /quit to exit, /reset to start fresh, or Ctrl-D twice to exit.\n")

        consecutive_eof = 0
        
        try:
            while True:
                try:
                    user_input = self._get_input()
                    consecutive_eof = 0  # Reset EOF counter on successful input
                    
                    if user_input is None:
                        break
                    
                    # Handle reset command
                    if user_input.strip() == '/reset':
                        self.reset_conversation()
                        continue
                    
                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    
                    # Send message with timeout handling
                    while True:
                        try:
                            assistant_message = self.send_message(user_input)
                            formatted_message = self._format_response(assistant_message)
                            print(f"\n{Colors.CLAUDE}Claude:{Colors.RESET} {formatted_message}\n")
                            break
                        except anthropic.APITimeoutError:
                            print("\nRequest timed out after 60 seconds.")
                            retry: str = input(
                                "Continue waiting? (y/n): "
                            ).strip().lower()
                            if retry not in ['y', 'yes']:
                                print("Exiting...")
                                return
                            print("Retrying...")
                        
                except EOFError:
                    consecutive_eof += 1
                    if consecutive_eof >= 2:
                        print("\nGoodbye!")
                        break
                    print()  # Add newline after first Ctrl-D
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
    
    def close(self) -> None:
        """Close the conversation and save according to storage mode."""
        if self.storage_mode == "none" or not self.transcript:
            return
        
        if self.storage_mode == "prompts":
            self._save_prompts_only()
        elif self.storage_mode == "full":
            self._save_full_conversation()
    
    def _save_full_conversation(self) -> None:
        """Save the complete conversation to JSON file."""
        model_short: str = (
            self.model.split('-')[1] if '-' in self.model else self.model
        )
        filename: str = f"{self.session_timestamp}_{model_short}_chat.json"
        
        conversations_dir: Path = (
            Path(__file__).parent / ".." / "conversations"
        )
        conversations_dir.mkdir(exist_ok=True)
        
        conversation_data: Dict[str, any] = {
            "session_start": self.session_timestamp,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "transcript": self.transcript
        }
        
        filepath: Path = conversations_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Conversation saved to: {filepath.name}")
    
    def _save_prompts_only(self) -> None:
        """Save only user inputs and system prompt to JSON file."""
        model_short: str = (
            self.model.split('-')[1] if '-' in self.model else self.model
        )
        filename: str = f"{self.session_timestamp}_{model_short}_prompts.json"
        
        conversations_dir: Path = (
            Path(__file__).parent / ".." / "conversations"
        )
        conversations_dir.mkdir(exist_ok=True)
        
        user_prompts: List[str] = [
            msg["content"] for msg in self.transcript if msg["role"] == "user"
        ]
        
        prompts_data: Dict[str, any] = {
            "session_start": self.session_timestamp,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompts": user_prompts
        }
        
        filepath: Path = conversations_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prompts saved to: {filepath.name}")


def main() -> None:
    """Main entry point for the Python code companion."""
    args: Dict[str, any] = docopt(__doc__)
    api_key: str = load_api_key()
    client: anthropic.Anthropic = anthropic.Anthropic(
        api_key=api_key,
        timeout=60.0
    )

    # Load system prompt if specified
    system_prompt: Optional[str] = load_system_prompt(args['--system'])
    
    model: str = args['--model']
    max_tokens: int = int(args['--max-tokens'])
    verbose: bool = args['--verbose']
    
    # Create and run conversation
    conversation: Conversation = Conversation(
        client=client,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        storage_mode="full"
    )
    conversation.verbose = verbose
    
    conversation.interact()
    conversation.close()


if __name__ == '__main__':
    main()
