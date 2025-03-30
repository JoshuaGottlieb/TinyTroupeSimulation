import os
from together import Together

client = Together(api_key = os.environ["TOGETHER_API_KEY"])

def call_llama(prompt: str, history: list[tuple[str, str]] = [], role_context: str = "") -> str:
    """
    Generates a response from a Llama model based on the given prompt.

    Args:
        prompt (str): The input prompt to send to the Llama model.
        history (list[tuple[str, str]], optional): A list of past interactions as (user_message, model_response) tuples.
                    Defaults to an empty list.
        role_context (str, optional): Additional role-based context to guide the model's response.
                    Defaults to an empty string.

    Returns:
        str: The generated response from the Llama model.
    """

    messages = []

    # If role context is provided, apply role context as a system message
    if role_context:
        messages.append({"role": "system", "content": role_context})

    # Define the prompt as a user message
    messages.append({"role": "user", "content": prompt})

    # If chat history is provided, apply history as messages using the appropriate roles
    if history:
        for role, content in history:
            messages.append({"role": role, "content": content})

    # Create a completion request with the prompt
    response = client.chat.completions.create(
        # Use the Llama-3-8b-chat-hf model
        model = "meta-llama/Llama-3-8b-chat-hf",
        messages = messages,
        temperature = 0.7,
    )

    # Return the content of the first response message
    return response.choices[0].message.content