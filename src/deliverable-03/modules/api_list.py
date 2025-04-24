# Absolute imports
import requests

# Relative imports
from .dataholder import *
from .evaluate import *
from .helper import add_to_event_stream, print_to_stream, process_api_call, register_function
from .model import *
from .preprocess import *
from .serialize import *

@register_function("api_help")
def api_help(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
             response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Displays a summary of all available API functions, including their name, prerequisites, 
    trigger phrases, description, and user input expectations.

    This function helps the user understand what the agent is capable of doing, 
    and how to invoke specific tasks based on trigger phrases or function descriptions.

    Args:
        dataholder (DataHolder): The shared object containing the current dataset and metadata.
        event_stream (List[Dict[str, Any]]): A list tracking the sequence of messages exchanged during the session.
        response (Dict[str, Any]): A dictionary for returning response metadata, typically modified during execution.

    Returns:
        Dict[str, Any]: The updated response dictionary, marked with success.
    """
    
    # Notify the user that help is being displayed
    print_to_stream(event_stream, role = "bot",
                    message = "Here is a list of all of the functions that I am capable of performing. "
                            "If a function has prerequisites, they will be shown.")

    # Loop through each API function in the metadata
    for api_function in dataholder.metadata:
        # Function name
        function_name = f"Function: {api_function}"

        # Prerequisite function, if any
        function_prerequisite = "Prerequisite: "
        if isinstance(dataholder.metadata[api_function]["prerequisite"], dict):
            function_prerequisite += dataholder.metadata[api_function]["prerequisite"].get("function", "None")
        else:
            function_prerequisite += "None"

        # Trigger phrases (showing up to 3 examples)
        function_triggers = f"Sample Trigger Phrases: {', '.join(dataholder.metadata[api_function]['trigger_word'][:3])}"

        # Function description
        function_description = f"Description: {dataholder.metadata[api_function]['description']['function']}"

        # Expected user inputs
        function_inputs = f"User Inputs: {dataholder.metadata[api_function]['description']['input']}"

        # Construct and print the help message
        message = "\n\n".join([
            function_name,
            function_prerequisite,
            function_triggers,
            function_description,
            function_inputs
        ])
        print_to_stream(event_stream, role = "bot", message = message)

    # Mark help function as successful in the response
    add_to_event_stream(event_stream, response, success = True)

    return response

@register_function("clear_history")
def clear_history(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                  response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clears all stored data and model state from the session, except for persistent metadata.

    This function resets the internal state of the `DataHolder` object by deleting all dynamically added attributes,
    with the exception of metadata (e.g., project settings, session config). It is typically used to reset the
    environment before starting a new task or model pipeline.

    Args:
        dataholder (DataHolder): Object that stores the current session’s data, model, features, and results.
        event_stream (List[Dict[str, Any]]): A list of messages exchanged during the workflow.
        response (Dict[str, Any]): Response object to store the outcome of the clear operation.

    Returns:
        Dict[str, Any]: A dictionary with success status and confirmation message.
    """
    # Clear all dynamic attributes from the DataHolder except metadata
    dataholder.delete_all_attributes(exceptions = ["metadata"])

    # Inform the user that the session state has been reset
    print_to_stream(event_stream, role = "bot", message = "All data cleared and ready for new function calls.")

    # Mark the operation as successful and return response
    add_to_event_stream(event_stream, response, success = True)
    return response

@register_function("select_model")
def select_model(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                 response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guides the user through selecting an appropriate modeling approach—classification or regression—
    based on their target variable and use case.

    If no target variable is found, it triggers the data loading API to ensure the data is ready.
    Once the target is available, the function explains the differences between classification and regression,
    prompts the user for a choice, and dispatches the relevant report-generating API.

    Args:
        dataholder (DataHolder): Object managing current session state, including user data and parameters.
        event_stream (List[Dict[str, Any]]): Stream of events used for UI interaction and logging.
        response (Dict[str, Any]): Initial response dictionary to be populated with results or error details.

    Returns:
        Dict[str, Any]: Updated response dictionary containing the result of the selected modeling path.
    """
    # Begin with a general introduction to the modeling decision process
    print_to_stream(event_stream, role = "bot",
                    message = "I will help you decide whether classification or regression is best for your data.")

    # Check whether a target variable ('y') is already stored in memory
    print_to_stream(event_stream, role = "bot", message = "Checking to see if your data is loaded.")
    y = dataholder.get("y")

    # If no target variable exists, trigger the CSV data loading and target selection routine
    if not y:
        print_to_stream(event_stream, role = "bot",
                        message = "No data is loaded, calling load_csv_and_select_target to load your data.")
        csv_response = process_api_call("load_csv_and_select_target", event_stream, dataholder)
        add_to_event_stream(event_stream, csv_response)

        # If loading fails, exit the function early
        if not csv_response.get("success", ""):
            print_to_stream(event_stream, role = "bot",
                            message = "Unable to load data, exiting.")
            add_to_event_stream(event_stream, response)
            return response

    # Notify the user of the detected or loaded target variable
    print_to_stream(event_stream, role = "bot", message = f"Your target variable is {dataholder.y.name}.")

    # Present a brief explanation of classification vs regression
    modeling_message = '\n'.join([
        "There are two types of modeling: classification and regression.",
        "Classification is when trying to predict a category or label (ex: Cats vs. Dogs).",
        "Regression is when trying to predict within a continuous range of numbers (ex: Housing Prices)."
    ])
    print_to_stream(event_stream, role = "bot", message = modeling_message)

    user_response = ""

    # Prompt the user to select the modeling type until a valid input is given
    while user_response not in ["c", "r"]:
        if "exit" in user_response:
            print_to_stream(event_stream, role = "bot", message = "User exit.")
            add_to_event_stream(event_stream, response)
            return response

        print_to_stream(event_stream, role = "bot", message = "Enter R for regression or C for classification.")
        user_response = input("Enter R or C. Enter EXIT to quit.").lower()
        print_to_stream(event_stream, role = "user", message = user_response)

    # User chose regression modeling
    if user_response == "r":
        print_to_stream(event_stream, role = "bot",
                        message = "I will create a regression report for you, follow the rest of the instructions.")
        return process_api_call("regression_report", event_stream, dataholder)

    # User chose classification modeling
    elif user_response == "c":
        print_to_stream(event_stream, role = "bot",
                        message = "I will create a classification report for you, follow the rest of the instructions.")
        return process_api_call("classification_report", event_stream, dataholder)

@register_function("load_csv_and_select_target")
def load_csv_and_select_target(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                               response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads a CSV into the DataHolder, normalizes and converts column types,
    prompts the user to select a target variable, and updates the event stream.

    Args:
        dataholder (DataHolder): Object responsible for managing and validating the dataset.
        event_stream (List[Dict[str, Any]]): The list that stores all response events.
        response (Dict[str, Any]): The response dictionary to be updated and returned.

    Returns:
        Dict[str, Any]: The response dictionary indicating the result of the operation.
    """
    # Load the CSV into the DataHolder
    if dataholder.load_dataframe():
        print_to_stream(event_stream, role = "bot", message = "Dataframe successfully loaded.")
    else:
        print_to_stream(event_stream, role = "bot",
                        message = f"Dataframe was unable to be loaded at: {dataholder.csv_path}")
        add_to_event_stream(event_stream, response)
        return response

    # Normalize column names for consistency
    if dataholder.normalize_column_names():
        print_to_stream(event_stream, role = "bot", message = "Columns successfully normalized.")
    else:
        print_to_stream(event_stream, role = "bot", message = "Unable to normalize column names.")

    # Attempt to convert column types
    if dataholder.convert_column_types():
        print_to_stream(event_stream, role = "bot", message = "Columns successfully converted.")
    else:
        print_to_stream(event_stream, role = "bot", message = "Columns were not able to be successfully converted.")

    # Present valid columns to the user
    valid_columns = [column.lower() for column in dataholder.df.columns.tolist()]
    print_to_stream(event_stream, role = "bot",
                    message = f"The possible columns are:\n{valid_columns}")

    # Prompt user to select the target column
    target_column = ""
    while target_column not in valid_columns:
        target_column = input("Please select a column to use as the target variable.").lower()
        print_to_stream(event_stream, role = "user", message = target_column)
        if "exit" in target_column:
            print_to_stream(event_stream, role = "bot", message = "Target column not selected, exiting.")
            add_to_event_stream(event_stream, response)
            return response

    # Attempt to split features and set target
    if dataholder.split_features(y = target_column):
        print_to_stream(event_stream, role = "bot", message = f"{target_column} set as the target column.")
        
        # Mark the operation as successful
        add_to_event_stream(event_stream, response, success = True)
        return response
    else:
        print_to_stream(event_stream, role = "bot", message = f"Unable to set {target_column} as the target column.")

        # Mark the operation as unsuccessful
        add_to_event_stream(event_stream, response)
        return response

@register_function("perform_regression")
def perform_regression(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                       response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs regression analysis on the dataset, including validation, preprocessing,
    model fitting, and handling of user input in case of task misclassification.

    Args:
        dataholder (DataHolder): Object responsible for managing and processing the dataset.
        event_stream (List[Dict[str, Any]]): The list that stores all response events.
        response (Dict[str, Any]): The response dictionary to be updated and returned.

    Returns:
        Dict[str, Any]: The response dictionary indicating the result of the regression process.
    """
    # Set the task to regression
    dataholder.set_machine_learning_task(regression_flag = True)
    preprocessor = DataFramePreprocessor(dataholder)

    # Validate if the target variable is suitable for regression
    correct_flag = preprocessor.validate_target_variable()

    if correct_flag == -1:
        # Handle case where target variable is incorrectly classified as a regression target
        user_response = ""
        print_to_stream(event_stream, role = "bot", message = "Identified task as classification, not regression.")

        while user_response not in ["y", "n"]:
            if "exit" in user_response:
                print_to_stream(event_stream, role = "bot", message = "User exit.")
                add_to_event_stream(event_stream, response)
                return response

            # Provide the user with the option to perform classification instead or reselect target
            print_to_stream(event_stream, role = "bot",
                            message = f"The specified task is regression (predicting a continuous value),"
                            + f" but the target column {dataholder.y.name} is a categorical column.")
            print_to_stream(event_stream, role = "bot",
                            message = "Regression is not possible on a categorical value."
                            +"\nWould you like to perform classification (predicting a category) instead?"
                            +"\nEnter Y to perform classification or N to reselect the target column.")
            user_response = input("Enter Y or N.").lower()
            print_to_stream(event_stream, role = "user", message = user_response)

        # Handle user decision: perform classification
        if user_response == "y":
            print_to_stream(event_stream, role = "bot", message = "User chose to execute perform_classification.")
            return process_api_call("perform_classification", event_stream, dataholder)

        # Handle user decision: reselect target column
        elif user_response == "n":
            print_to_stream(event_stream, role = "bot", message = "User chose to reselect target variable.")
            csv_response = process_api_call("load_csv_and_select_target", event_stream, dataholder)
            add_to_event_stream(event_stream, csv_response)

            if csv_response.get("success", ""):
                print_to_stream(event_stream, role = "bot",
                                message = "Target variable successfully reselected. Retrying perform_regression.")
                return process_api_call("perform_regression", event_stream, dataholder)
            else:
                print_to_stream(event_stream, role = "bot",
                                message = "Unable to reselect target variable or reload csv.")
                add_to_event_stream(event_stream, response)
                return response

    # Proceed with preprocessing if the target is valid
    print_to_stream(event_stream, role = "bot", message = "Preprocessing dataframe")
    if not preprocessor.preprocess():
        print_to_stream(event_stream, role = "bot", message = "Failed to preprocess dataframe, exiting.")
        add_to_event_stream(event_stream, response)
        return response

    print_to_stream(event_stream, role = "bot", message = "Dataframe successfully preprocessed.")
    print_to_stream(event_stream, role = "bot", message = "Fitting model(s), this may take a while.")

    # Fit and score the regression model
    modeler = AutomaticModeler(dataholder)
    if not modeler.fit_and_score_model():
        print_to_stream(event_stream, role = "bot", message = "Failed to fit and score model, exiting.")
        add_to_event_stream(event_stream, response)
        return response

    # Successful model fitting and scoring
    print_to_stream(event_stream, role = "bot", message = "Successfully built model.")
    add_to_event_stream(event_stream, response, success = True)
    return response

@register_function("perform_classification")
def perform_classification(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                           response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a classification workflow by validating the target variable, preprocessing the data,
    and training a classification model. Includes logic to handle cases where the data might be better suited
    for regression or discretization.

    Args:
        dataholder (DataHolder): Object that contains the dataframe, target, features, and metadata.
        event_stream (List[Dict[str, Any]]): A list of messages exchanged during the pipeline.
        response (Dict[str, Any]): Response object to store the result of the classification task.

    Returns:
        Dict[str, Any]: A dictionary with success status and any redirection to other functions.
    """

    # Set the task type to classification
    dataholder.set_machine_learning_task(regression_flag = False)
    preprocessor = DataFramePreprocessor(dataholder)

    # Validate if the target is categorical (suitable for classification)
    correct_flag = preprocessor.validate_target_variable()

    # If the target is continuous (better for regression)
    if correct_flag == 1:
        user_response = ""
        print_to_stream(event_stream, role = "bot", message = "Identified task as regression, not classification.")

        # Ask user if they want to switch to regression
        while user_response not in ["y", "n"]:
            if "exit" in user_response:
                print_to_stream(event_stream, role = "bot", message = "User exit.")
                add_to_event_stream(event_stream, response)
                return response

            print_to_stream(event_stream, role = "bot",
                            message = f"The specified task is classification (predicting a category), "
                            + f"but the target column {dataholder.y.name} is a continuous column.")
            print_to_stream(event_stream, role = "bot",
                            message = "Would you like to perform a regression (predicting a continuous value) instead?"
                            +"\nEnter Y to perform regression or N to continue.")
            user_response = input("Enter Y or N.").lower()
            print_to_stream(event_stream, role = "user", message = user_response)

        # Redirect to regression if user agrees
        if user_response == "y":
            print_to_stream(event_stream, role = "bot", message = "User chose to execute perform_regression.")
            return process_api_call("perform_regression",event_stream, dataholder)

        # Ask if user wants to discretize the target variable
        elif user_response == "n":
            user_response = ""
            while user_response not in ["y", "n"]:
                if "exit" in user_response:
                    print_to_stream(event_stream, role = "bot", message = "User exit.")
                    add_to_event_stream(event_stream, response)
                    return response

                print_to_stream(event_stream, role = "bot",
                                message = f"Would you like to discretize (create categories) for the target column "
                                + f"{dataholder.y.name}?")
                print_to_stream(event_stream, role = "bot",
                                message = "Enter Y to discretize or N to reselect the target column.")
                user_response = input("Enter Y or N.").lower()
                print_to_stream(event_stream, role = "user", message = user_response)

            if user_response == "y":
                # Ask for number of bins (categories)
                print_to_stream(event_stream, role = "bot",
                                message = "Select the number of categories to create or enter nothing for default (3 bins).")
                bins = input("Enter the number of categories to use (default 3).")
                try:
                    bins = int(bins)
                    if bins <= 0:
                        bins = 3
                except:
                    bins = 3
                print_to_stream(event_stream, role = "user", message = str(bins))
                print_to_stream(event_stream, role = "bot", message = f"User chose to discretize data into {bins} bins.")
                interval_labels = preprocessor.bin_continuous_data(num_bins = bins)
                if interval_labels:
                    print_to_stream(event_stream, role = "bot", message = "\n".join(interval_labels))
                else:
                    print_to_stream(event_stream, role = "bot", message = "Binning failed.")
                    add_to_event_stream(event_stream, response)
                    return response
            elif user_response == "n":
                # Reselect target column
                print_to_stream(event_stream, role = "bot", message = "User chose to reselect target variable.")
                csv_response = process_api_call("load_csv_and_select_target", event_stream, dataholder)
                add_to_event_stream(event_stream, csv_response)

                if csv_response.get("success", ""):
                    print_to_stream(event_stream, role = "bot",
                                    message = "Target variable successfully reselected. Retrying perform_classification.")
                    return process_api_call("perform_classification", event_stream, dataholder)
                else:
                    print_to_stream(event_stream, role = "bot",
                                    message = "Unable to reselect target variable or reload CSV.")
                    add_to_event_stream(event_stream, response)
                    return response

    # Preprocess the data before modeling
    print_to_stream(event_stream, role = "bot", message = "Preprocessing dataframe")
    if not preprocessor.preprocess():
        print_to_stream(event_stream, role = "bot", message = "Failed to preprocess dataframe, exiting.")
        add_to_event_stream(event_stream, response)
        return response

    print_to_stream(event_stream, role = "bot", message = "Dataframe successfully preprocessed.")
    print_to_stream(event_stream, role = "bot", message = "Fitting model(s), this may take a while.")

    # Train and score classification model
    modeler = AutomaticModeler(dataholder)
    if not modeler.fit_and_score_model():
        print_to_stream(event_stream, role = "bot", message = "Failed to fit and score model, exiting.")
        add_to_event_stream(event_stream, response)
        return response

    # Final success response
    print_to_stream(event_stream, role = "bot", message = "Successfully built model.")
    add_to_event_stream(event_stream, response, success = True)
    return response

@register_function("regression_report")
def regression_report(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                      response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a regression report by sending data to a FastAPI endpoint and optionally saves it as a PDF.

    Args:
        dataholder (DataHolder): Object containing dataset and configuration for the regression report.
        event_stream (List[Dict[str, Any]]): Stream of messages for tracking progress and interaction logs.
        response (Dict[str, Any]): Initial response object to be updated with the result.

    Returns:
        Dict[str, Any]: The updated response dictionary with success status or error messages.
    """

    # URL of the local FastAPI endpoint that generates the regression report
    url = "http://localhost:8000/regression_report/"

    # Notify the user that data preparation has started
    print_to_stream(event_stream, role = "bot", message = "Preparing data for report.")
    payload = serialize_dataholder(dataholder)

    # Notify the user that the report generation request is being sent
    print_to_stream(event_stream, role = "bot", message = "Creating report. This may take a while.")
    post_response = requests.post(url, json = payload)

    # Handle API response
    if post_response.status_code == 200:
        report = post_response.json()["content"]
    else:
        print(post_response.json()["error"])
        return response

    # If configured to save the report as a PDF
    if dataholder.save_pdf:
        if not dataholder.save_path:
            dataholder.save_path = "report.pdf"  # Use default path if none specified
        save_status = serialized_regression_report_to_pdf(report, output_path = dataholder.save_path)
        if save_status:
            print_to_stream(event_stream, role = "bot", message = f"Successfully saved pdf at {dataholder.save_path}")
        else:
            print_to_stream(event_stream, role = "bot", message = f"Unable to save pdf at {dataholder.save_path}")
            add_to_event_stream(event_stream, response)
            return response
    else:
        # Display the report in serialized format (e.g., to a notebook or UI)
        display_serialized_regression_report(report)

    # Update the response and return it
    add_to_event_stream(event_stream, response, success=True)
    return response

@register_function("classification_report")
def classification_report(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                          response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a classification report by sending data to a FastAPI endpoint and optionally saves it as a PDF.

    Args:
        dataholder (DataHolder): Object containing dataset and configuration for the classification report.
        event_stream (List[Dict[str, Any]]): Stream of messages for tracking progress and interaction logs.
        response (Dict[str, Any]): Initial response object to be updated with the result.

    Returns:
        Dict[str, Any]: The updated response dictionary with success status or error messages.
    """

    # URL of the local FastAPI endpoint that generates the classification report
    url = "http://localhost:8000/classification_report/"

    # Notify the user that data preparation has started
    print_to_stream(event_stream, role="bot", message="Preparing data for report.")
    payload = serialize_dataholder(dataholder)

    # Notify the user that the report generation request is being sent
    print_to_stream(event_stream, role="bot", message="Creating report. This may take a while.")
    post_response = requests.post(url, json=payload)

    # Handle API response
    if post_response.status_code == 200:
        report = post_response.json()["content"]
    else:
        return response  # Return unchanged response if the request fails

    # If configured to save the report as a PDF
    if dataholder.save_pdf:
        if not dataholder.save_path:
            dataholder.save_path = "report.pdf"  # Use default path if none specified
        save_status = serialized_classification_report_to_pdf(report, output_path=dataholder.save_path)
        if save_status:
            print_to_stream(event_stream, role="bot", message=f"Successfully saved pdf at {dataholder.save_path}")
        else:
            print_to_stream(event_stream, role="bot", message=f"Unable to save pdf at {dataholder.save_path}")
            add_to_event_stream(event_stream, response)
            return response
    else:
        # Display the report in serialized format (e.g., to a notebook or UI)
        display_serialized_classification_report(report)

    # Update the response and return it
    add_to_event_stream(event_stream, response, success=True)
    return response