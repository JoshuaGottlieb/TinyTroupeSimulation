from .helper import register_function, add_to_event_stream, print_to_stream, process_api_call
from .dataholder import *
from .preprocess import *
from .evaluate import *
from .model import *
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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
    Evaluates a trained regression model using statistical analysis, diagnostic visualizations,
    and a natural language explanation generated by a language model.
    
    Args:
        dataholder (DataHolder): Object that contains the dataframe, target, features, model predictions,
            and performance metrics for training and test sets.
        event_stream (List[Dict[str, Any]]): A list of messages exchanged during the regression pipeline.
        response (Dict[str, Any]): Response object used to store and propagate function results.

    Returns:
        Dict[str, Any]: A dictionary containing the success status and updated response content.
    """
    # Alert user that the bot is generating statistics
    print_to_stream(event_stream, role = "bot", message = "Generating regression statistics and residuals.")
    
    # Create a model evaluator to access regression utilities
    evaluator = ModelEvaluator(dataholder)

    # Compute regression summary stats and feature coefficients
    coefficient_df, f_stat, f_p_value, aic, bic = evaluator.regression_statistics()

    # Generate and display the ANOVA summary table for the model
    anova = evaluator.plot_anova(coefficient_df, f_stat, f_p_value, aic, bic)

    # Compute residuals (true - predicted) for the test set
    residuals = dataholder.y_test - dataholder.predictions['test']

    # Visualize residuals vs. predicted values to diagnose model error behavior
    residual_plot = evaluator.plot_residuals(residuals)

    # Alert the user that the bot is generating explanation
    print_to_stream(event_stream, role = "bot", message = "Generating AI explanation.")

    # Build prompt for the language model to generate a plain-language performance summary
    prompt = f"""
    You are an expert data scientist tasked with evaluating a regression model using the following information:
    
    - **Target Variable**: {dataholder.y.name}
    - **Training R²**: {dataholder.regression_scores["train"]["r2_score"]:.2f}
    - **Testing R²**: {dataholder.regression_scores["test"]["r2_score"]:.2f}
    - **Training RMSE**: {dataholder.regression_scores["train"]["rmse"]:.2g}
    - **Testing RMSE**: {dataholder.regression_scores["test"]["rmse"]:.2g}
    - **ANOVA table** (formatted as a string):
    {anova}
    - **Top 5 significant features**:
    {coefficient_df.head(5).to_string()}
    - **Predictions** (a list of predicted values for the test set):
    {dataholder.predictions["test"]}
    - **Residuals** (calculated as the difference between actual and predicted values):
    {residuals}
    
    ---
    
    ### Instructions:
    
    - Use **Markdown format** with all section headers in ALL CAPS using header level 5 (`#####`).
    - Each header should be followed by a **line break**, then the content should begin on the next line.
    - Write in an **active voice**, with a clear and engaging tone.
    - All technical terms must be explained in **plain English** as if the reader has **no background** in statistics or ML.
    - Prioritize clarity, context, and interpretation.
    - Round all numeric values to **2 decimal places**, use **commas as thousands separators**, and for values less than 0.001 or greater than 10,000, use **scientific notation**.
    - Help the reader **understand how to evaluate the model**, not just state metrics.
    - When discussing features, mention **coefficient** and **p-value** (if available), and explain the real-world implication of each feature.
    
    ---
    
    ### Required Sections:
    
    ##### PREDICTION TARGET  
    Explain what the model is predicting, based on the target variable `{dataholder.y.name}`.  
    Why is this prediction useful or important?
    
    ##### METRIC EXPLANATIONS  
    Explain what **R²** (coefficient of determination) and **RMSE** (root mean squared error) represent.  
    Clarify what high or low values imply in terms of model performance.  
    Define **overfitting** and **underfitting**, and describe how the difference between training and testing scores reveals these problems.
    
    ##### OVERFITTING OR UNDERFITTING?  
    Use the provided scores to determine if the model is overfitting, underfitting, or well-generalized.  
    Support your answer with numerical comparisons and interpretation.
    
    ##### ANOVA TABLE OVERVIEW  
    Explain what each ANOVA column means:  
      - Degrees of Freedom  
      - F-statistic  
      - p-value  
    Then interpret the ANOVA table as a whole: Does it suggest the model has statistically significant predictive power?
    
    ##### SIGNIFICANT FEATURES ANALYSIS  
    List and explain the top 5 most significant features in the format:  
    "**Feature Name** (coefficient = X.XX, p-value = Y.YY): Explanation."  
    Clarify what it means for a feature to be significant.  
    Note: One-hot encoded features (e.g., containing underscores and capital letters or numbers) are binary and should be interpreted as on/off flags.
    
    ##### RESIDUAL PLOT EVALUATION  
    Explain how a residual plot is used to evaluate model performance.  
    Discuss the following characteristics:  
      - **Randomness**: Are residuals randomly scattered around zero?  
      - **Patterns**: Do residuals form any systematic shape?  
      - **Homoscedasticity**: Is the variance constant?  
      - **Outliers**: Are there any extreme residuals that suggest poor predictions?  
      - **Normality**: Do residuals roughly follow a normal distribution?
    
    ##### KEY INSIGHTS FROM RESIDUALS AND PREDICTIONS  
    Summarize key findings based on residuals and predictions:  
      - Are residuals evenly distributed?  
      - Are there signs of heteroscedasticity or other patterns?  
      - Are the predictions accurate, or do they show large errors?  
      - Do the residuals suggest systematic problems?
    
    ##### MODEL RATING  
    Provide a final rating in this format:  
    **"X / 10"**  
    Explain your score based on:  
      - Model fit  
      - Generalization performance  
      - Statistical significance of features  
      - Prediction quality and residual behavior
    """

    # Generate and display the Markdown summary using the language model
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)

    # Save the PDF report if user has requested it
    if dataholder.save_pdf:
        # Use default save path if an empty string was provided
        if not dataholder.save_path:
            dataholder.save_path = "report.pdf"
        save_status = evaluator.save_regression_report(
            anova,
            residual_plot,
            judge_eval,
            output_path = dataholder.save_path
        )

        plt.close('all')

        # Report success or failure to save PDF
        if save_status:
            print_to_stream(event_stream, role = "bot", message = f"Successfully saved pdf at {dataholder.save_path}")
        else:
            print_to_stream(event_stream, role = "bot", message = f"Unable to save pdf at {dataholder.save_path}")
            add_to_event_stream(event_stream, response)
            return response
    else:
        # Show evaluation report inline if PDF saving is not enabled
        print_to_stream(event_stream, role = "bot", message = anova)
        plt.show()
        display(Markdown(judge_eval))

    # Prompt user whether to clear the dataholder for further processing
    print_to_stream(event_stream, role = "bot",
                    message = "Would you like to clear stored data? Enter Y or N. (Defaults to N)")

    user_response = input("Enter Y or N.").lower()

    # If specified, clear the event history from the dataholder
    if user_response == "y":
        add_to_event_stream(event_stream, {
                        "role": "detect_api",
                        "content": "clear_history",
                        "success": True
        })
    
        clear_history_response = {"role": "api_call", "content": "clear_history", "success": False}
        clear_history_response = clear_history(dataholder, event_stream, clear_history_response)
        add_to_event_stream(event_stream, clear_history_response)
    
    # Update and return final response
    add_to_event_stream(event_stream, response, success = True)
    return response

@register_function("classification_report")
def classification_report(dataholder: DataHolder, event_stream: List[Dict[str, Any]],
                          response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a classification evaluation workflow by computing performance metrics, visualizing confusion matrices,
    and generating a comprehensive model analysis using a language model.

    Args:
        dataholder (DataHolder): Object that contains the dataframe, target variable, model predictions,
            and performance metrics for both training and testing sets.
        event_stream (List[Dict[str, Any]]): A list of messages exchanged during the classification pipeline.
        response (Dict[str, Any]): A response dictionary used to return success status and results.

    Returns:
        Dict[str, Any]: A dictionary containing the success flag and updated response content.
    """
    # Alert user that the bot is generating visuals
    print_to_stream(event_stream, role = "bot", message = "Generating report visuals, this may take some time.")
    
    # Instantiate the evaluator to access classification metrics, visualizations, and SHAP utilities
    evaluator = ModelEvaluator(dataholder)

    # Generate a confusion matrix using test predictions
    test_confusion_matrix = confusion_matrix(
        dataholder.y_test,
        dataholder.predictions["test"]
    )

    # Plot both train and test confusion matrices
    confusion_matrices = evaluator.plot_confusion_matrices()

    # Retrieve model metadata
    model_name = evaluator.model_name
    target_variable = dataholder.y.name

    # Generate ROC curve plot and AUC score
    roc_curve, roc_auc_score = evaluator.plot_roc_curve()

    # Generate SHAP summary plots and retrieve top SHAP features
    shap_plots, top_avg_scores, top_max_scores = evaluator.shap_summary_plot()

    # Format SHAP values into readable strings for the report prompt
    average_shap_values_string = '\n'.join([
        f"Feature: {feature}, Score: {score}" for feature, score in top_avg_scores
    ])
    max_shap_values_string = '\n'.join([
        f"Feature: {feature}, Score: {score}" for feature, score in top_max_scores
    ])

    # Alert the user that the bot is generating explanation
    print_to_stream(event_stream, role = "bot", message = "Generating AI explanation.")

    # Compose prompt for LLM-based model evaluation
    prompt = f"""
    You are a senior machine learning analyst. Evaluate the performance of a classification model using the information provided below.
    
    ### Input Information:
    - **Model type**: {model_name}
    - **Target variable**: {target_variable}
    - **Confusion Matrix (rows = true labels, columns = predicted labels)**:  
    {test_confusion_matrix}
    
    - **ROC AUC score**: {roc_auc_score:.2f}
    - **Average SHAP values per feature**:  
    {average_shap_values_string}
    
    - **Maximum SHAP value per feature**:  
    {max_shap_values_string}
    
    - **Classification scores**:  
    - **Training accuracy**: {dataholder.classification_scores['train']['accuracy']:.2f}
    - **Testing accuracy**: {dataholder.classification_scores['test']['accuracy']:.2f}
    - **Training precision**: {dataholder.classification_scores['train']['precision']:.2f}
    - **Testing precision**: {dataholder.classification_scores['test']['precision']:.2f}
    - **Training recall**: {dataholder.classification_scores['train']['recall']:.2f}
    - **Testing recall**: {dataholder.classification_scores['test']['recall']:.2f}
    - **Training F1 score**: {dataholder.classification_scores['train']['f1_score']:.2f}
    - **Testing F1 score**: {dataholder.classification_scores['test']['f1_score']:.2f}
    ---
    
    ### Instructions:
    - Use Markdown format with section headers in all caps using header level 5 ("#####").
    - Begin each section on a new line.
    - Use **active voice** and explain technical terms in **plain, accessible language** for non-experts.
    - Round all numeric values to **2 decimal places** and use **commas for thousands separators**.
    - Explain how to interpret each metric and visual (e.g., confusion matrix, ROC curve, SHAP values).
    - Use examples or analogies where helpful.
    
    ---
    
    ### Required Sections:
    
    **PREDICTION GOAL**  
    Briefly explain what the model is predicting based on the target variable `{target_variable}`.  
    Explain why it's important to get this prediction right.
    
    **OVERFITTING OR UNDERFITTING**  
    Compare training and testing accuracy, precision, recall, and F1 scores.  
    Explain whether the model is overfitting, underfitting, or performing well in generalization.  
    Highlight any major discrepancies.
    
    **CONFUSION MATRIX INTERPRETATION**  
    Interpret the confusion matrix:  
      - Explain true positives, false positives, false negatives, and true negatives.  
      - Mention if there’s class imbalance or a tendency to favor one class.  
      - Point out if certain types of errors (e.g., false negatives) are more critical in this context.  
    If it is a **multi-class confusion matrix**, walk through the matrix row by row or class by class to identify
    which classes are well-predicted and which are confused with others.
    
    **ROC CURVE & AUC SCORE**  
    Explain what the ROC curve represents, and what the AUC (Area Under the Curve) score means.  
    Give a plain-English interpretation of the model’s ability to distinguish between classes based on the ROC AUC score.
    
    **SHAP VALUE INTERPRETATION**  
    You are given the average and maximum SHAP values for each feature.  
    Explain:  
      - What SHAP values mean in terms of feature importance  
      - What the **average SHAP value** tells us (overall importance across all predictions)  
      - What the **maximum SHAP value** reveals (how influential a feature can be in a single prediction)  
    Identify the top 3–5 features by SHAP values and describe how they influence predictions.
    
    **KEY INSIGHTS & RECOMMENDATIONS**  
    Summarize the model’s overall behavior and performance.  
    Highlight anything that might require attention (e.g., bias, weak generalization, heavy reliance on one feature).  
    Offer suggestions for improving model performance (e.g., collecting more data, rebalancing classes, tuning thresholds).
    
    **MODEL RATING**  
    Give a rating in this format:  
    **"Rating (X / 10)"**  
    Base the score on generalization performance, interpretability, and reliability of the model's decisions.
    """

    # Generate human-readable model evaluation using LLM
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)

    # Save the PDF report if user has requested it
    if dataholder.save_pdf:
        # Use default save path if an empty string was provided
        if not dataholder.save_path:
            dataholder.save_path = "report.pdf"
        save_status = evaluator.save_classification_report(
            confusion_matrices,
            roc_curve,
            shap_plots,
            judge_eval,
            output_path = dataholder.save_path
        )

        plt.close('all')

        # Report success or failure to save PDF
        if save_status:
            print_to_stream(event_stream, role = "bot", message = f"Successfully saved pdf at {dataholder.save_path}")
        else:
            print_to_stream(event_stream, role = "bot", message = f"Unable to save pdf at {dataholder.save_path}")
            add_to_event_stream(event_stream, response)
            return response
    else:
        # Show evaluation report inline if PDF saving is not enabled
        plt.show()
        display(Markdown(judge_eval))

    # Prompt user whether to clear the dataholder for further processing
    print_to_stream(event_stream, role = "bot",
                    message = "Would you like to clear stored data? Enter Y or N. (Defaults to N)")

    user_response = input("Enter Y or N.").lower()

    # If specified, clear the event history from the dataholder
    if user_response == "y":
        add_to_event_stream(event_stream, {
                        "role": "detect_api",
                        "content": "clear_history",
                        "success": True
        })
    
        clear_history_response = {"role": "api_call", "content": "clear_history", "success": False}
        clear_history_response = clear_history(dataholder, event_stream, clear_history_response)
        add_to_event_stream(event_stream, clear_history_response)
    
    # Update and return final response
    add_to_event_stream(event_stream, response, success = True)
    return response