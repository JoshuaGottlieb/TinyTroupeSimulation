import os
from together import Together
from typing import Any

class LlamaBot:
    def __init__(self, api_key: str = os.environ["TOGETHER_API_KEY"],
                 role_context: str = "", temperature: float = 0.7):
        """
        Initializes the ChatBot with the given API key and role context.

        Args:
            api_key (str): The API key for accessing Together services.
            role_context (str): An optional role to assign as context for the ChatBot.
            temperature (float): The temperature to use for the Llama model.
                                    Higher temperatures are more creative, while lower temperatures are more technical.
        """
        self.client = Together(api_key = api_key)
        self.model = "meta-llama/Llama-3-8b-chat-hf"
        self.temperature = temperature
        self.role = {"role": "system", "content": role_context}
        self.clear_history()

    def call_llama(self, prompt: str, save_history: bool = True) -> str:
        """
        Generates a response from a Llama model based on the given prompt.
    
        Args:
            prompt (str): The input prompt to send to the Llama model.
            save_history (bool): Whether to save the prompt and response to the ChatBot's history.
    
        Returns:
            str: The generated response from the Llama model.
        """

        prompt_dict = {"role": "user", "content": prompt}
        
        # Generate response
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.history + [prompt_dict],
            temperature = self.temperature
        ).choices[0].message.content

        # If enabled, save the prompt and the response to the history
        if save_history:
            self.history.append(prompt_dict)
            self.history.append({
                "role": "assistant",
                "content": response
            })
            
        return response

    def call_llama_eval(self, prompt: str) -> Any:
        return eval(self.call_llama(prompt = prompt, save_history = False))

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

        # Prepend history with role context
        if self.role["content"]:
            self.history.append(self.role)
        
        return

    def get_role(self) -> str:
        """
        Returns the role context of the chatbot.

        Returns:
            str: The role context.
        """
        return self.role["content"]

    def set_role(self, role_context: str, overwrite: bool = False) -> None:
        """
        Sets the role context of the chatbot.

        Args:
            role_context (str): A role to assign as context for the ChatBot.
            overwrite (bool): Whether to overwrite the existing role context for the ChatBot.
        """
        # Overwrites or sets the role context
        if overwrite or not self.role["content"]:
            print(f"Role context is now \"{role_context}\"")
            self.role = {"role": "system", "content": role_context}
            
            # Inserts the role context into the history as needed
            if self.history and self.history[0]["role"] == "system":
                self.history[0] = self.role
            else:
                self.history.insert(0, self.role)
        # If the overwrite flag is not passed and role context already exists, alert the user
        else:
            print(f"Role context already exists: \"{self.get_role()}\"")
            print("Role will not be replaced, pass in overwrite = True if you wish to overwrite the existing role context.")

        return