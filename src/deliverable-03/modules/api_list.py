from .helper import register_function, add_to_event_stream, print_to_stream
from .dataholder import *
from .preprocess import *
from .evaluate import *
from .model import *
from sklearn.metrics import confusion_matrix
import numpy as np

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
                print_to_stream(event_stream, role="bot", message="User exit.")
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
            response["content"] = "perform_classification"
            add_to_event_stream(event_stream, {
                "role": "detect_api",
                "content": "perform_classification",
                "success": True
            })
            return perform_classification(dataholder, event_stream, response)

        # Handle user decision: reselect target column
        elif user_response == "n":
            print_to_stream(event_stream, role = "bot", message = "User chose to reselect target variable.")
            add_to_event_stream(event_stream, {
                "role": "detect_api",
                "content": "load_csv_and_select_target",
                "success": True
            })
            csv_response = {"role": "api_call", "content": "load_csv_and_select_target", "success": False}
            csv_response = load_csv_and_select_target(dataholder, event_stream, response)
            add_to_event_stream(event_stream, csv_response)

            if csv_response.get("success", ""):
                print_to_stream(event_stream, role = "bot",
                                message = "Target variable successfully reselected. Retrying perform_regression.")
                return perform_regression(dataholder, event_stream, response)
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
            response["content"] = "perform_regression"
            add_to_event_stream(event_stream, {
                "role": "detect_api",
                "content": "perform_regression",
                "success": True
            })
            return perform_regression(dataholder, event_stream, response)

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
                add_to_event_stream(event_stream, {
                    "role": "detect_api",
                    "content": "load_csv_and_select_target",
                    "success": True
                })

                csv_response = {"role": "api_call", "content": "load_csv_and_select_target", "success": False}
                csv_response = load_csv_and_select_target(dataholder, event_stream, response)
                add_to_event_stream(event_stream, csv_response)

                if csv_response.get("success", ""):
                    print_to_stream(event_stream, role = "bot",
                                    message = "Target variable successfully reselected. Retrying perform_classification.")
                    add_to_event_stream(event_stream, response)
                    return perform_classification(dataholder, event_stream, response)
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
    # Create a model evaluator to access regression utilities
    evaluator = ModelEvaluator(dataholder)

    # Compute regression summary stats and feature coefficients
    coefficient_df, f_stat, f_p_value, aic, bic = evaluator.regression_statistics()

    # Generate and display the ANOVA summary table for the model
    anova = evaluator.plot_anova(coefficient_df, f_stat, f_p_value, aic, bic)
    print_to_stream(event_stream, role = "bot", message = anova)

    # Compute residuals (true - predicted) for the test set
    residuals = dataholder.y_test - dataholder.predictions['test']

    # Visualize residuals vs. predicted values to diagnose model error behavior
    residual_plot = evaluator.plot_residuals(residuals)
    plt.show()

    # Build prompt for the language model to generate a plain-language performance summary
    prompt = f"""
    You are an expert data scientist tasked with evaluating a regression model using the following information:

    - **Target Variable**: {dataholder.y.name}
    - **Training R²**: {dataholder.scores["train"]["r2_score"]:.2f}
    - **Testing R²**: {dataholder.scores["test"]["r2_score"]:.2f}
    - **Training RMSE**: {dataholder.scores["train"]["rmse"]:.2g}
    - **Testing RMSE**: {dataholder.scores["test"]["rmse"]:.2g}
    - **ANOVA table** (formatted as a string):
    {anova}
    - **Top 5 significant features**:
    {coefficient_df.head(5).to_string()}
    - **Predictions** (a list of predicted values for the test set):
    {dataholder.predictions["test"]}
    - **Residuals** (calculated as the difference between actual values and predicted values):
    {residuals}

    ---
    
    ### Instructions:
    
    - Use Markdown format with section headers in all caps using header level 5 ("#####").
    - Begin each section on a new line. Do not write content on the same line as the header.
    - Use **active voice** and explain all technical terms in **plain, accessible language** for a reader with no
      background in statistics or machine learning.
    - Focus especially on explaining what **R²**, **RMSE**, **overfitting**, **underfitting**, **residual plots**, and
      each term in the ANOVA table mean.
    - Explain how to interpret the differences between training and testing scores.
    - Round all numeric values to **2 decimal places**, and use **commas for thousands separators**.
      For values < 0.001 or > 10,000, use **scientific notation**.
    - Interpret how well the model is generalizing based on the scores.
    - When discussing features, mention the **coefficient** and **p-value** (if provided),
      and explain what a significant feature means in context.

    ---
    
    ### Required Sections:

    **PREDICTION TARGET**  
    Briefly explain what the model is trying to predict, based on the target variable `{dataholder.y.name}`.  
    Describe in simple terms what this variable represents and why it might be important to forecast it accurately.

    **METRIC EXPLANATIONS**  
    Explain what R² (coefficient of determination) and RMSE (root mean squared error) measure.  
    Describe what high or low values mean and how to interpret them in a regression context.  
    Explain overfitting and underfitting in simple terms, and describe how the difference between
    training and testing scores helps detect these.

    **OVERFITTING OR UNDERFITTING?**  
    Compare training and testing R² and RMSE.  
    Explain whether the model is overfitting, underfitting, or performing appropriately.  
    Support your conclusion with the numerical values and differences.

    **ANOVA TABLE OVERVIEW**  
    Explain what each column in the ANOVA table means
    (e.g., degrees of freedom, sum of squares, mean square, F-statistic, p-value).  
    Interpret the table as a whole: What does it tell us about the overall significance of the model?

    **SIGNIFICANT FEATURES ANALYSIS**  
    List and explain the top 5 most significant features using this format:  
    "Feature Name (coefficient = X.XX, p-value = Y.YY): Explanation of its importance and how it influences the target."  
    For one-hot encoded features (e.g., with underscores and capital letters or numbers),
    explain that they are binary indicators.

    **RESIDUAL PLOT EVALUATION**  
    Explain how to evaluate the residual plot and what insights it provides.  
    Describe the significance of the following characteristics:  
      - **Randomness**: Residuals should appear randomly scattered around zero if the model is a good fit.  
      - **Patterns**: A pattern in the residuals (e.g., a curve or systematic structure) suggests the model is not
        capturing some underlying relationship in the data.  
      - **Homoscedasticity**: Residuals should have constant variance (not funnel-shaped or widening/narrowing).  
      - **Outliers**: Look for residuals that are far from the zero line. Outliers may indicate influential data points
        or errors in predictions.  
      - **Normality**: Ideally, residuals should follow a normal distribution (though this is not a strict requirement).

    **KEY INSIGHTS FROM RESIDUALS AND PREDICTIONS**  
    Provide a summary of the **residuals** and **predictions** with the following insights:  
      - **Spread of Residuals**: Are the residuals randomly distributed around zero or
        do they show any specific pattern (e.g., a funnel shape, curve, etc.)?  
      - **Outliers**: Are there any significant outliers in the residuals that could indicate
        problematic predictions or data points?  
      - **Variance**: Do the residuals seem to have constant variance (homoscedasticity),
        or do they exhibit increasing or decreasing variance (heteroscedasticity)?  
      - **Prediction Accuracy**: How close are the residuals to zero? Does this indicate
        good prediction accuracy or suggest areas for improvement?

    **MODEL RATING**  
    Give an overall performance rating using this format:  
    **"X / 10"**  
    Base your score on model fit, generalization, and statistical significance of features.  
    Briefly explain your reasoning.
    """

    # Generate and display the Markdown summary using the language model
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)
    display(Markdown(judge_eval))

    # Mark function completion successfully in the event stream
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
    # Create a model evaluator for classification utilities and visualization
    evaluator = ModelEvaluator(dataholder)

    # Compute confusion matrix on the test set
    test_confusion_matrix = confusion_matrix(
        dataholder.y_test,
        dataholder.predictions["test"]
    )

    # Plot and display the training and testing confusion matrices
    confusion_matrices = evaluator.plot_confusion_matrices();
   

    # Construct a detailed evaluation prompt for the language model
    prompt = f"""
    You are an expert data scientist tasked with evaluating the results of a classification model.
    
    Use the following performance metrics and confusion matrix data to assess how well the model performs and
    whether it is overfitting:
    
    - **Training accuracy**: {dataholder.scores['train']['accuracy']}
    - **Testing accuracy**: {dataholder.scores['test']['accuracy']}
    - **Training precision**: {dataholder.scores['train']['precision']}
    - **Testing precision**: {dataholder.scores['test']['precision']}
    - **Training recall**: {dataholder.scores['train']['recall']}
    - **Testing recall**: {dataholder.scores['test']['recall']}
    - **Training F1 score**: {dataholder.scores['train']['f1_score']}
    - **Testing F1 score**: {dataholder.scores['test']['f1_score']}
    - **Testing confusion matrix (rows = true labels, columns = predicted labels)**:  
    {test_confusion_matrix}

    ---
    
    ### Formatting Instructions:

    - Use Markdown format with section headers in all caps using header level 5 ("#####").
    - Start each section on a new line. Do not write content on the same line as the header.
    - Explain what **accuracy**, **precision**, **recall**, and **F1 score** mean,
        including how macro-averaging works in a multi-class setting.
    - Explain what a **confusion matrix** represents in the context of multi-class classification.
    - Interpret the confusion matrix: Identify which classes are being confused most often,
        and discuss what that might imply.
    - Use **active voice** and **layman-friendly language** throughout.
    - Round all numeric values to **2 decimal places**.
    - Focus on explaining whether the model is **overfitting**, **underfitting**, or performing well by comparing
        training vs. testing scores.

    ---
    
    ### Required Sections:

    **METRIC EXPLANATIONS**  
    Explain in simple terms what accuracy, precision, recall, and F1 score each measure.  
    Clarify that **macro averaging** treats each class equally, regardless of class imbalance.  
    Also define what a multi-class confusion matrix is and how it should be interpreted.

    **CONFUSION MATRIX INSIGHTS**  
    Use the matrix to identify which specific classes the model most frequently misclassifies.  
    Describe whether these misclassifications are serious or acceptable depending on how far off the
    predicted classes are from the true ones.  
    If applicable, note whether certain classes are consistently underrepresented in predictions.

    **OVERFITTING EVALUATION**  
    Compare the training and testing scores across all metrics.  
    If the model performs significantly better on the training set than the test set, it may be overfitting.  
    Explain your conclusion clearly using the provided numbers.

    **MODEL RATING**  
    Provide an overall rating of the model’s performance using this format:  
    **"X / 10"**  
    Base your score on generalization ability, class balance, and how well the model performs across metrics.  
    Be fair and explain your reasoning briefly.
    """

    # Generate the LLM-based model evaluation
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)

    # Save the PDF if flag is set
    if dataholder.save_pdf:
        save_status = evaluator.save_classification_report(
            confusion_matrices,
            judge_eval,
            output_path = dataholder.save_path
        )
        # Print whether the report was successfully saved to a PDF
        if save_status:
            print_to_stream(event_stream, role = "bot", message = f"Successfully saved pdf at {dataholder.save_path}")
        else:
            print_to_stream(event_stream, role = "bot", message = f"Unable to save pdf at {dataholder.save_path}")
            add_to_event_stream(event_stream, response)
            return response
    # Otherwise, display report inline
    else:
        plt.show()
        display(Markdown(judge_eval))
        
    # Finalize the response
    add_to_event_stream(event_stream, response, success = True)
    return response