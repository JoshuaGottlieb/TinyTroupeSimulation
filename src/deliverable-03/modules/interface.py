import json
import os
from .api_list import *
from .helper import *
from .chatbot import *
from .dataholder import *

# Path to metadata file containing dataset structure, label types, etc.
metadata_filepath = os.path.join(os.path.dirname(__file__), "metadata.json")

class ModelBot:
    """
    A conversational assistant that interprets user input to perform automated
    machine learning tasks (classification or regression). The bot utilizes a
    language model to detect user intent and interactively guides the modeling process.

    Attributes:
        bot (LlamaBot): The chatbot interface for conversational responses.
        dataholder (DataHolder): Holds and manages user data and metadata.
        event_stream (List[Dict]): Stores the conversation history for context tracking.
    """

    def __init__(self, role: str = "You are a helpful agent."):
        """
        Initializes the ModelBot with a language model, an empty DataHolder, 
        and loads metadata necessary for task inference and feature validation.

        Args:
            role (str): Describes the bot’s role/persona in the conversation.
        """
        self.bot = LlamaBot(role_context = role, temperature = 0.7)
        self.dataholder = DataHolder()
        self.dataholder.metadata = load_metadata(metadata_filepath)
        self.event_stream = []

    def chat(self):
        """
        Starts an interactive loop with the user to perform ML tasks.
        The bot listens for prompts, tries to detect and process API calls, 
        or responds with a default LLM-generated answer.

        Loop exits when user inputs 'exit'.
        """
        # Welcome and system messages
        print_to_stream(self.event_stream, role = "system",
                        message = "This is a helper agent designed to automatically perform basic modeling tasks for you.")
        print_to_stream(self.event_stream, role = "system",
                        message = "This agent is powered by Llama 3 and is not a replacement for a human data scientist.")
        print_to_stream(self.event_stream, role = "system",
                        message = "To get a list of functions available to this agent, type \"help\".")
        print_to_stream(self.event_stream, role = "system",
                        message = "If you want to restart the chat, type \"clear data\".")
        print_to_stream(self.event_stream, role = "system",
                        message = "You can exit the chat at any time by typing EXIT.")

        print_to_stream(self.event_stream, role = "bot", message = "How can I help you today?")

        # Initial prompt
        prompt = input("User: ").lower()
        print_to_stream(self.event_stream, role = "user", message = prompt)

        # Continue until user types "exit"
        while "exit" not in prompt:
            self.event_stream.append({"role": "user", "content": prompt})

            # Detect whether the user's prompt should trigger an API call
            api_call = detect_api_call(prompt, self.event_stream, self.dataholder)

            if api_call["success"]:
                api_name = api_call["content"]
                print_to_stream(self.event_stream, role = "bot",
                                message = f"Attempting to call API function {api_name}.")

                # Run the corresponding function
                api_result = process_api_call(api_name, self.event_stream, self.dataholder)
                if api_result["success"]:
                    print_to_stream(self.event_stream, role = "bot",
                                    message = f"Successfully processed API call. How else may I help you?")
                else:
                    print_to_stream(self.event_stream, role = "bot",
                                    message = f"Something went wrong trying to execute {api_name}, so please try again.")
            else:
                # No API detected; fallback to chatbot response
                print_to_stream(self.event_stream, role = "bot",
                                message = f"No API call detected, defaulting to LLM response.")
                bot_response = self.bot.call_llama(prompt = prompt, save_history = False)
                print_to_stream(self.event_stream, role = "bot", message = bot_response)

            # Next user input
            prompt = input("User: ").lower()
            print_to_stream(self.event_stream, role = "user", message = prompt)

        # End of session message
        print_to_stream(self.event_stream, role = "bot", message = "Thank you for using this helper agent!")