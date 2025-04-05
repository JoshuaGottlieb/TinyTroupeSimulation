import os
from together import Together
from typing import Any

class LlamaBot:
    def __init__(self, api_key: str = os.environ["TOGETHER_API_KEY"],
                 role_context: str = "", temperature: float = 0.7):
        """
        Initializes the ChatBot with the given API key, role context, and temperature for the Llama model.
    
        Args:
            api_key (str, optional): The API key for accessing Together services.
                            Defaults to the environment variable "TOGETHER_API_KEY".
            role_context (str, optional): A string representing the role context for the chatbot
                            (e.g., "You are a helpful assistant"). Defaults to an empty string.
            temperature (float, optional): A float representing the temperature for the Llama model.
                    Higher values make the model's responses more creative, while lower values make it more deterministic.
                    Defaults to 0.7.
        """
        self.client = Together(api_key = api_key)
        self.model = "meta-llama/Llama-3-8b-chat-hf"
        self.temperature = temperature
        self.role = {"role": "system", "content": role_context}
        self.clear_history()

    def call_llama(self, prompt: str, save_history: bool = True) -> str:
        """
        Generates a response from the Llama model based on the given prompt.
    
        Args:
            prompt (str): The input prompt to send to the Llama model.
            save_history (bool, optional): Whether to save the prompt and response to the ChatBot's history. 
                                           Defaults to True, which saves the history.
    
        Returns:
            str: The generated response from the Llama model.
        """
        # Prepare the prompt as a dictionary with role and content for the Llama model
        prompt_dict = {"role": "user", "content": prompt}
        
        # Generate the response from the Llama model by sending the current history and prompt
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.history + [prompt_dict], # Add the prompt to the history for context
            temperature = self.temperature
        ).choices[0].message.content  # Extract the content of the response
        
        # If save_history is True, append the prompt and response to the ChatBot's history
        if save_history:
            # Save user prompt and assistant's response to maintain conversation context
            self.history.append(prompt_dict)
            self.history.append({
                "role": "assistant",
                "content": response
            })
        
        return response

    def call_llama_eval(self, prompt: str) -> Any:
        """
        Evaluates the response from the Llama model based on the provided prompt.
    
        This method sends the prompt to the Llama model, receives the generated response,
        and evaluates it using the `eval()` function.
    
        Args:
            prompt (str): The input prompt to send to the Llama model for evaluation.
    
        Returns:
            Any: The result of evaluating the Llama model's response.
        """
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
        Sets or updates the role context for the ChatBot.
    
        This method allows you to specify a role for the ChatBot, which provides it with context.
        Example: "You are an assistant"
        
        If `overwrite` is set to True, it will replace any existing role context.
    
        Args:
            role_context (str): The role or context to assign to the ChatBot (e.g., "You are a helpful assistant").
            overwrite (bool, optional): Whether to overwrite the existing role context if one already exists.
                                        Defaults to False.
        """
        # If overwrite flag is True or no role context exists, set or update the role context
        if overwrite or not self.role["content"]:
            # Update the role context in the role dictionary
            self.role = {"role": "system", "content": role_context}

            print(f"Role context is now \"{role_context}\"")
            
            # Insert or update the role context in the history
            if self.history and self.history[0]["role"] == "system":
                self.history[0] = self.role # Replace the first entry in history with the new role context
            else:
                self.history.insert(0, self.role) # Add the role context to the beginning of the history
        else:
            # If the overwrite flag is False and a role context already exists, alert the user
            print(f"Role context already exists: \"{self.get_role()}\"")
            print("Role will not be replaced, pass in overwrite = True if you wish to overwrite the existing role context.")
    
        return