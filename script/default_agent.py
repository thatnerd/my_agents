#!/usr/bin/env python3
"""Claude Chat CLI

Usage:
    default_agent.py [--model=<model>] [--system=<file>] [--max-tokens=<tokens>]
    default_agent.py (-h | --help)

Options:
    -h --help              Show this screen.
    --model=<model>        Claude model to use [default: claude-3-haiku-20240307].
    --system=<file>        System prompt file in ../prompt/ directory.
    --max-tokens=<tokens>  Maximum tokens in response [default: 4096].

Commands:
    /exit, /quit           Exit the program.
    Ctrl-C, Ctrl-D twice   Exit the program.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import anthropic
from docopt import docopt


def load_api_key() -> str:
    """Load Anthropic API key from ~/.keys/anthropic_api_key."""
    key_path = Path.home() / ".keys" / "anthropic_api_key"
    return key_path.read_text().strip()


def save_conversation(messages: List[Dict[str, str]], model: str, system_prompt: Optional[str]) -> None:
    """Save conversation to JSON file in ../conversations/ directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split('-')[1] if '-' in model else model
    filename = f"{timestamp}_{model_short}_chat.json"
    
    conversations_dir = Path(__file__).parent / ".." / "conversations"
    conversations_dir.mkdir(exist_ok=True)
    
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "messages": messages
    }
    
    filepath = conversations_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversation saved to: {filepath.name}")


def load_system_prompt(filename: Optional[str]) -> Optional[str]:
    """Load system prompt from ../prompt/ directory."""
    if not filename:
        return None
    
    prompt_path = Path(__file__).parent / ".." / "prompt" / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    
    content = prompt_path.read_text().strip()
    return content if content else None


def main() -> None:
    """Main chat loop."""
    args = docopt(__doc__)
    
    # Load API key
    api_key = load_api_key()
    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=5.0
    )
    
    # Load system prompt if specified
    system_prompt = load_system_prompt(args['--system'])
    
    # Conversation history
    messages: List[Dict[str, str]] = []
    
    model = args['--model']
    max_tokens = int(args['--max-tokens'])
    
    print(f"Claude Chat CLI (Model: {model})")
    if system_prompt:
        print(f"System prompt loaded from: {args['--system']}")
    print("Type your message and press Enter. Use /exit, /quit, Ctrl-C, or Ctrl-D twice to exit.\n")
    
    consecutive_eof = 0
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                consecutive_eof = 0  # Reset EOF counter on successful input
                
                if user_input.lower() in ['/exit', '/quit']:
                    break
                
                if not user_input:
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages
                }
                
                if system_prompt:
                    request_params["system"] = system_prompt
                
                # Send request to Claude
                response = client.messages.create(**request_params)
                
                # Extract and display response
                assistant_message = response.content[0].text
                print(f"\nClaude: {assistant_message}\n")
                
                # Add Claude's response to conversation history
                messages.append({"role": "assistant", "content": assistant_message})
                
            except EOFError:
                consecutive_eof += 1
                if consecutive_eof >= 2:
                    print("\nGoodbye!")
                    break
                print()  # Add newline after first Ctrl-D
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    # Save conversation if there were any messages
    if messages:
        save_conversation(messages, model, system_prompt)


if __name__ == "__main__":
    main()
