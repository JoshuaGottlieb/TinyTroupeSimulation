# Absolute imports
import json
import re

# Third-party imports
from collections.abc import Callable

# Local imports
from .chatbot import *
from .dataholder import *

# Typing imports
from typing import Dict, Any, List

# Initialize function registry for API calls
function_registry = {}

def register_function(api_name: str):
    """
    Decorator to register API functions dynamically into the function registry.

    Args:
        api_name (str): The name of the API function to be registered in the function_registry.

    Returns:
        Callable: A decorator that registers the provided function under the given api_name.
    """

    def decorator(func: Callable[..., Dict[str, Any]]):
        """
        Registers the given function into the function_registry with the specified api_name.

        Args:
            func (Callable[..., Dict[str, Any]]): The function to be registered.

        Returns:
            Callable: The original function passed to the decorator.
        """
        function_registry[api_name] = func
        return func

    return decorator

def add_to_event_stream(event_stream: List[Dict[str, Any]], response: Dict[str, Any],
                        success: bool = False, show: bool = False) -> None:
    """
    Appends a response dictionary to the event stream and optionally marks it as successful
    and displays it.

    Args:
        event_stream (List[Dict[str, Any]]): The list that stores all response events.
        response (Dict[str, Any]): The response dictionary to be added to the event stream.
        success (bool, optional): If True, adds a "success" flag to the response. Defaults to False.
        show (bool, optional): If True, prints the response in a readable format. Defaults to False.
    """

    # Add the "success" flag to the response if applicable
    if success:
        response["success"] = success

    # Append the response to the event stream
    event_stream.append(response)

    # Print the response in a readable format if requested
    if show:
        print(response["role"].title() + ": " + response["content"])

    return

def print_to_stream(event_stream: List[Dict[str, Any]], role: str,
                    message: str, show = True) -> None:
    """
    Creates a response dictionary and adds it to the event stream. Optionally prints the message.

    Args:
        event_stream (List[Dict[str, Any]]): The list that stores all response events.
        role (str): The role of the sender (e.g., "user", "assistant").
        message (str): The message content to be added.
        show (bool, optional): If True, prints the message to the console. Defaults to True.
    """

    # Create the response dictionary
    response = {"role": role, "content": message}

    # Add the response to the event stream and optionally print it
    add_to_event_stream(event_stream, response, show=show)

    return


def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Loads and returns metadata from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing metadata.
    """

    # Open the file and load its contents as JSON
    with open(filepath, "r") as f:
        metadata = json.load(f)

    return metadata


def match_trigger_words(user_message: str, trigger_words: List[str]) -> bool:
    """
    Check if all words in each trigger phrase are present in the user's message.

    Args:
        user_message (str): The message input by the user.
        trigger_words (List[str]): A list of phrases to be matched against the user's message.

    Returns:
        bool: True if all words in at least one trigger phrase are found in the user's message.
    """
    # Clean the user message by removing punctuation and converting to lowercase
    cleaned_message = re.sub(r"[^\w\s]", "", user_message.lower())
    message_words = cleaned_message.split()  # Split message into individual words

    for trigger in trigger_words:
        cleaned_trigger = re.sub(r"[^\w\s]", "", trigger.lower())
        trigger_words_list = cleaned_trigger.split()  # Split trigger phrase into words

        # Check if all words in the trigger phrase are present in the user message
        if all(word in message_words for word in trigger_words_list):
            # print(f"Match found for trigger: '{trigger}'")
            return True

    # print("No match found.")
    return False

def detect_api_call(user_message: str, event_stream: List[Dict[str, Any]], dataholder: DataHolder) -> Dict[str, Any]:
    """
    Detects whether the user's message matches any API trigger words and updates the event stream accordingly.

    Args:
        user_message (str): The message input from the user.
        event_stream (List[Dict[str, Any]]): The list that stores all response events.
        dataholder (DataHolder): Object used to validate and store parameter values
                                    and containing metadata for all available APIs.

    Returns:
        Dict[str, Any]: A response dictionary indicating the detection result and matched API name, if any.
    """

    # Initialize the default response
    response = {
        "role": "detect_api",
        "content": "",
        "success": False
    }

    # Check user message against trigger words for each API
    for api_name, api_metadata in dataholder.metadata.items():
        trigger_words = api_metadata.get("trigger_word", [])

        if match_trigger_words(user_message, trigger_words):
            # If a match is found, log the intent and update the response
            print_to_stream(event_stream, role = "bot",
                            message = f"Intent detected for API call: {api_name}")
            response["content"] = api_name
            response["success"] = True
            break

    # Add the detection result to the event stream
    add_to_event_stream(event_stream = event_stream, response = response)

    return response

def process_api_call(api_name: str, event_stream: List[Dict[str, Any]], dataholder: DataHolder) -> Dict[str, Any]:
    """
    Executes a named API call by checking prerequisites, prompting for user inputs if needed,
    and invoking the corresponding function from the function registry.

    The function also handles fallback logic, such as:
    - Executing prerequisite APIs when data is missing.
    - Offering alternative APIs when the main call is infeasible.
    - Logging all relevant steps and responses to the event stream.

    Args:
        api_name (str): Name of the target API to execute.
        event_stream (List[Dict[str, Any]]): Stream of interaction messages to track all steps of the pipeline.
        dataholder (DataHolder): Object that stores data, tracks metadata, and validates API requirements.

    Returns:
        Dict[str, Any]: A response dictionary containing execution status and any result metadata.
    """

    # Initialize the response structure for the current API call
    response = {
        "role": "api_call",
        "content": api_name,
        "success": False
    }

    # Check whether the API is defined in the metadata
    if api_name not in dataholder.metadata:
        print_to_stream(event_stream, role = "bot",
                        message = f"Error: API {api_name} is not defined in the metadata.")
        add_to_event_stream(event_stream, response)
        return response

    # Extract and evaluate prerequisites for the API, if defined
    prerequisites = dataholder.metadata[api_name]["prerequisite"]
    if isinstance(prerequisites, dict):
        prerequisites = {key: eval(value) for key, value in prerequisites.get("data", {}).items()}
        missing_prereqs = dataholder.validate_parameters(prerequisites)

        # If required inputs are missing
        if missing_prereqs:
            # Check for alternative API if provided in metadata
            alternative = dataholder.metadata[api_name].get("alternative", "")
            if alternative:
                alternative_prereqs = {
                    key: eval(value) for key, value in
                    dataholder.metadata[alternative]["prerequisite"].get("data", {}).items()
                }
                missing_alternative_prereqs = dataholder.validate_parameters(alternative_prereqs)

                if not missing_alternative_prereqs:
                    # Offer to switch to alternative API
                    alternative_message = ' '.join([
                        f"Alternative function {alternative} found.",
                        "Do you want to try this function instead? Enter Y or N. Defaults to N."
                    ])
                    print_to_stream(event_stream, role = "bot", message = alternative_message)

                    user_confirmation = input("Enter Y or N: ").lower()
                    if user_confirmation == "y":
                        return process_api_call(alternative, event_stream, dataholder)

            # Execute prerequisite sub-API if alternative is not chosen
            sub_api = dataholder.metadata[api_name]["prerequisite"]["function"]
            reason = dataholder.metadata[api_name]["prerequisite"].get("reason", "")
            
            print_to_stream(event_stream, role = "bot",
                            message = f"Error: Missing prerequisites for API {api_name}, "
                                    f"invoking prerequisite API {sub_api}.")
            print_to_stream(event_stream, role="bot",
                            message = f"Reason for missing prerequisites: {reason}")
            add_to_event_stream(event_stream, response)

            # Log and invoke the sub-API to satisfy missing data
            add_to_event_stream(event_stream, {
                "role": "detect_api",
                "content": sub_api,
                "success": True
            })

            sub_response = process_api_call(sub_api, event_stream, dataholder)
            add_to_event_stream(event_stream, sub_response)

            # Retry original API if prerequisite was fulfilled
            if sub_response["success"]:
                return process_api_call(api_name, event_stream, dataholder)
            else:
                return sub_response

    # Look up and execute the actual registered API function
    api_function = function_registry.get(api_name)

    if api_function:
        # Prompt user for any missing inputs required by this API
        for key, datatype in dataholder.metadata[api_name]["sample_payload"].items():
            while not dataholder.validate_parameter(key, eval(datatype)):
                # Notify user that an input is required and suggest typing 'HELP' for guidance
                print_to_stream(event_stream, role = "bot",
                                message = "Missing inputs needed, please enter inputs. (Enter HELP for description.)")
                value = input(f"Please enter {key} (datatype: {datatype}): ")

                # If user enters 'HELP', show detailed input instructions from metadata
                if value.lower() == "help":
                    help_description = dataholder.metadata[api_name]["description"]["input"]
                    print_to_stream(event_stream, role = "bot",
                                    message = f"Here are the instructions for the inputs:\n{help_description}")
                    continue

                # Log user input and attempt to cast it to the expected datatype
                print_to_stream(event_stream, role = "user", message = f"{key} = {value}")
                try:
                    if eval(datatype) == bool:
                        dataholder.set(key, bool(int(value))) # Accepts 0/1 or other int-like bool inputs
                    else:
                        dataholder.set(key, eval(datatype)(value))
                except:
                    # Silently retry on failed type conversion
                    continue

        # Call the API function with validated inputs
        response = api_function(dataholder, event_stream, response)

        # If API call fails, notify user and clear state
        if not response["success"]:
            error_message = dataholder.metadata[api_name].get("error_message", "")
            if error_message:
                print_to_stream(event_stream, role = "bot", message = error_message)
            clear_history = process_api_call("clear_history", event_stream, dataholder)

        return response

    else:
        # Error: No matching function found in the registry
        print_to_stream(event_stream, role = "bot",
                        message = f"Error: No function found for API call {api_name}")
        return response