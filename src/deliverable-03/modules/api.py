# Absolute imports
import base64
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix
from typing import Any, Dict, List, Optional

# Relative imports
from .dataholder import DataHolder
from .evaluate import ModelEvaluator
from .serialize import serialize_classification_report, serialize_regression_report

# Initialize the FastAPI application
app = FastAPI()

class DataHolderInput(BaseModel):
    """
    Pydantic model for receiving and validating input data required for 
    classification or regression reports via the FastAPI endpoint.

    Attributes:
        X_train (List[Dict[str, Any]]): Training feature data.
        X_test (List[Dict[str, Any]]): Testing feature data.
        y_train (List[Any]): Training labels/targets.
        y_test (List[Any]): Testing labels/targets.
        predictions (Dict[str, List[Any]]): Dictionary of training and testing predictions.
        best_model (str): Base64-encoded string of a pickled model object.
        target_variable (str): Name of the target variable being predicted.
        classification_scores (Optional[Dict[str, Dict[str, float]]]): 
            Optional evaluation scores for classification models.
        regression_scores (Optional[Dict[str, Dict[str, float]]]): 
            Optional evaluation scores for regression models.
    """
    X_train: List[Dict[str, Any]]
    X_test: List[Dict[str, Any]]
    y_train: List[Any]
    y_test: List[Any]
    predictions: Dict[str, List[Any]]
    best_model: str  # base64-encoded string of pickled model
    target_variable: str
    classification_scores: Optional[Dict[str, Dict[str, float]]] = None
    regression_scores: Optional[Dict[str, Dict[str, float]]] = None

def deserialize_model(b64_model: str) -> Any:
    """
    Decodes and unpickles a base64-encoded machine learning model.

    Args:
        b64_model (str): A base64-encoded string representing a pickled model object.

    Returns:
        Any: The deserialized model object.

    Raises:
        ValueError: If decoding or unpickling fails.
    """
    try:
        # Decode the base64 string and unpickle the model
        pickled_model = base64.b64decode(b64_model.encode('utf-8'))
        model = pickle.loads(pickled_model)
        return model
    except Exception as e:
        raise ValueError(f"Failed to deserialize model: {e}")

@app.post("/regression_report/")
def regression_report(data: DataHolderInput) -> Dict[str, Any]:
    """
    Endpoint to evaluate a trained regression model and generate an interpretable report.

    This function deserializes the trained model, reconstructs the data,
    performs statistical analysis (e.g., ANOVA, residual analysis), and invokes
    a language model to generate a Markdown-formatted report in plain English.

    Args:
        data (DataHolderInput): Input object containing training/testing data, predictions,
            model (base64-encoded), target variable name, and regression scores.

    Returns:
        Dict[str, Any]: A dictionary with status code and either a generated report (`content`)
        or an error message (`error`).
    """
    try:
        # Step 1: Decode the base64-encoded pickled model
        model = deserialize_model(data.best_model)

        # Step 2: Convert raw input lists of dicts into pandas DataFrames and Series
        X_train = pd.DataFrame(data.X_train)
        X_test = pd.DataFrame(data.X_test)
        y_train = pd.Series(data.y_train, name = data.target_variable)
        y_test = pd.Series(data.y_test, name = data.target_variable)

        # Step 3: Rebuild DataHolder object with processed data
        dataholder = DataHolder()
        dataholder.X_train = X_train
        dataholder.X_test = X_test
        dataholder.y_train = y_train
        dataholder.y_test = y_test
        dataholder.predictions = {k: np.array(v) for k, v in data.predictions.items()}
        dataholder.best_model = model
        dataholder.regression_scores = data.regression_scores

    except Exception as e:
        # Return a 500 response if deserialization or data parsing fails
        return JSONResponse(status_code = 500, content = {"error": str(e)})

    # Initialize the evaluator object for regression diagnostics and visuals
    evaluator = ModelEvaluator(dataholder)

    # Generate core regression statistics: coefficients, F-test, AIC/BIC
    coefficient_df, f_stat, f_p_value, aic, bic = evaluator.regression_statistics()

    # Generate ANOVA summary as formatted Markdown
    anova = evaluator.plot_anova(coefficient_df, f_stat, f_p_value, aic, bic)

    # Calculate residuals (true - predicted values) for test set
    residuals = dataholder.y_test.to_numpy() - dataholder.predictions['test']

    # Create a residuals plot to analyze error patterns
    residual_plot = evaluator.plot_residuals(residuals)

    # Construct a detailed prompt for the LLM to generate an interpretive summary
    prompt = f"""
    You are an expert data scientist tasked with evaluating a regression model using the following information:
    
    - **Target Variable**: {dataholder.y_test.name}
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
    Explain what the model is predicting, based on the target variable `{dataholder.y_test.name}`.  
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

    # Use the language model to generate a human-readable evaluation summary
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)

    try:
        # Serialize plots and language model summary into a unified report
        report = serialize_regression_report(anova, residual_plot, judge_eval)

        return JSONResponse(status_code = 200, content = {"content": report})

    except Exception as e:
        # Handle serialization/formatting failures
        return JSONResponse(status_code = 500, content = {"error": str(e)})

@app.post("/classification_report/")
def classification_report(data: DataHolderInput) -> Dict[str, Any]:
    """
    Endpoint to evaluate a classification model and generate a comprehensive, interpretable report.

    This function reconstructs input data, computes classification metrics, visualizes key results 
    (confusion matrices, ROC curve, SHAP values), and uses a language model to create a plain-language 
    summary of the model's performance.

    Args:
        data (DataHolderInput): Input object containing training/testing data, model predictions, 
            performance metrics, target variable, and a base64-encoded model.

    Returns:
        Dict[str, Any]: A dictionary containing either a successfully generated report under the "content" key,
        or an error message under the "error" key.
    """
    try:
        # Step 1: Decode and load the pickled model from base64
        model = deserialize_model(data.best_model)

        # Step 2: Convert incoming data to pandas-compatible structures
        X_train = pd.DataFrame(data.X_train)
        X_test = pd.DataFrame(data.X_test)
        y_train = pd.Series(data.y_train, name = data.target_variable)
        y_test = pd.Series(data.y_test, name = data.target_variable)

        # Step 3: Rebuild a DataHolder object with structured inputs
        dataholder = DataHolder()
        dataholder.X_train = X_train
        dataholder.X_test = X_test
        dataholder.y_train = y_train
        dataholder.y_test = y_test
        dataholder.predictions = {k: np.array(v) for k, v in data.predictions.items()}
        dataholder.best_model = model
        dataholder.classification_scores = data.classification_scores
    except Exception as e:
        # Catch errors related to model deserialization or data conversion
        return JSONResponse(status_code = 500, content = {"error": str(e)})

    # Instantiate evaluator for classification-specific analysis
    evaluator = ModelEvaluator(dataholder)

    # Generate confusion matrix using test data
    test_confusion_matrix = confusion_matrix(
        dataholder.y_test,
        dataholder.predictions["test"]
    )

    # Generate and return both train and test confusion matrix plots
    confusion_matrices = evaluator.plot_confusion_matrices()

    # Retrieve metadata for prompt generation
    model_name = evaluator.model_name
    target_variable = dataholder.y_test.name

    # Create ROC curve and compute AUC score
    roc_curve, roc_auc_score = evaluator.plot_roc_curve()

    # Create SHAP plots and extract top feature importance scores
    shap_plots, top_avg_scores, top_max_scores = evaluator.shap_summary_plot()

    # Format SHAP values into readable strings for the LLM prompt
    average_shap_values_string = '\n'.join([
        f"Feature: {feature}, Score: {score}" for feature, score in top_avg_scores
    ])
    max_shap_values_string = '\n'.join([
        f"Feature: {feature}, Score: {score}" for feature, score in top_max_scores
    ])

    # Compose natural language prompt to evaluate classification performance
    prompt = f"""
    You are a senior machine learning analyst. Evaluate the performance of a classification model using the information provided below.
    
    ### Input Information:
    - **Model type**: {model_name}
    - **Target variable**: {target_variable}
    - **Confusion Matrix (top left: TN, top right: FP, bottom left: FN, bottom right: TP)**:  
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
      - Explain the purpose of precision, recall, accuracy, and F1 scores.
      - Point out which of precision, recall, accuracy, and F1 scores may be more critical in this context.
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

    # Use language model to create human-readable summary of model performance
    judge_eval = evaluator.judge.call_llama(prompt = prompt, save_history = False)

    try:
        # Combine visualizations and summary text into a final report format
        report = serialize_classification_report(confusion_matrices, roc_curve, shap_plots, judge_eval)

        return JSONResponse(status_code = 200, content = {"content": report})
    except Exception as e:
        # Catch serialization or formatting errors
        return JSONResponse(status_code = 500, content = {"error": str(e)})