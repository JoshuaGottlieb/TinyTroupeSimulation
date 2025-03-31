import re
import subprocess
import os
import streamlit as st
from together import Together
from typing import Optional, Any

class LlamaBot:
    def __init__(self, api_key: str = os.environ["TOGETHER_API_KEY"], role_context: str = ""):
        """
        Initializes the ChatBot with the given API key and role context.

        Args:
            api_key (str): The API key for accessing Together services.
            role_context (str): An optional role to assign as context for the ChatBot.
        """
        self.client = Together(api_key = api_key)
        self.model = "meta-llama/Llama-3-8b-chat-hf"
        self.history = []
        self.temperature = 0.7
        self.role = {"role": "system", "content": role_context}

    def call_llama(self, prompt: str) -> str:
        """
        Generates a response from a Llama model based on the given prompt.
    
        Args:
            prompt (str): The input prompt to send to the Llama model.
    
        Returns:
            str: The generated response from the Llama model.
        """
        # Add the prompt to the history
        self.history.append({
            "role": "user",
            "content": prompt
        })

        # Generate response
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.history,
            temperature = self.temperature
        ).choices[0].message.content

        # Add the response to history
        self.history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def get_history(self) -> list:
        """
        Returns the conversation history.

        Returns:
            list: The conversation history.
        """
        return self.history

    def clear_history(self) -> None:
        """
        Clears the conversation history.
        """
        self.history = []
        return

def isolate_python_code(input_string: str) -> Optional[str]:
    """
    Extracts Python code from a given input string.

    This function looks for a block of text that is enclosed within triple backticks (```),
    specifically targeting blocks that start with ```python.

    Args:
        input_string (str): The input string containing the Python code block.

    Returns:
        Optional[bool]: Returns the extracted Python code if found, else returns None.
    """
    # Define the pattern to match the Python code block
    pattern = r"```(?:\w+)?\n(.*?)\n```"

    # Search for the pattern in the input string
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        # Extract the Python code
        return match.group(1)
    else:
        # Return None if no Python code block is found
        return None

@st.fragment
def save_python_scripts(code_snippets: list[str]) -> bool:
    """
    Displays a text input for the user to enter a file path and provides a download button 
    to save the generated Python script.

    Args:
        code_snippets (list[str]): A list of Python code snippets to be saved.

    Returns:
        bool: True if the download button is clicked, otherwise False.
    """

    # Prompt the user to enter a file path for saving the script
    save_path = st.text_input("Enter a file path to save the generated code")
    
    if save_path:
       return st.download_button(
           label = f"Download the script to {save_path}.py",
           data = '\n\n'.join(code_snippets),
           file_name = f"{save_path}.py",
           mime = "text/plain"
       )
