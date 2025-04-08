from .chatbot import *
from .dataholder import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Any, Dict, Tuple
from IPython.display import display, Markdown

class ModelEvaluator():
    def __init__(self, dataholder: DataHolder):
        """
        Initializes the model evaluation and scoring pipeline with training and testing data.
    
        This constructor initializes the model, stores the dataset (both features and target), and sets up
        the necessary data for evaluating the model’s performance. The pipeline for preprocessing steps is 
        also created from the provided model.
    
        Args:
            X_train (pd.DataFrame): The feature matrix for training.
            X_test (pd.DataFrame): The feature matrix for testing.
            y_train (pd.Series): The target variable for training.
            y_test (pd.Series): The target variable for testing.
            model (Any): A scikit-learn pipeline or model with preprocessing and estimator steps.
            task_type (bool): Type of task; 0 for classification, 1 for regression.
            predictions (Dict[str, Dict[str, np.array]]): A dictionary to store predictions on train and test data.
            scores (Dict[str, Dict[str, np.array]]): A dictionary to store evaluation scores (e.g., accuracy, RMSE)
                    on train and test data.
        """
        self.dataholder = dataholder
        
        # Store the model and model type (estimator part of the pipeline)
        self.model_name, self.model = self.dataholder.best_model.named_steps['estimator'].steps[0]
        self.transformer = Pipeline(self.dataholder.best_model.steps[:-1])
        self.transformer.fit(self.dataholder.X_train)
        
        # LLM-based evaluation agent for interpreting model results in plain language
        self.judge = LlamaBot(
            role_context = """
                You are an AI assistant designed to evaluate the results of models.
                Your answers should provide explanations and numerical ratings on a scale from 1–10.
                Your explanations should be catered towards a non-technical audience.
            """,
            temperature = 0.2
        )
    
    def regression_statistics(self) -> Tuple[pd.DataFrame, float, float]:
        """
        Calculates coefficient statistics (t-statistics, p-values) and the F-statistic
        for regression models like LinearRegression, Lasso, or ElasticNet.
    
        Returns:
            Tuple:
                - results_df (pd.DataFrame): DataFrame containing the coefficient statistics with columns 
                  'coefficient', 'std_Error', 't_stat', and 'p_value'.
                - f_stat (float): The overall F-statistic for the model.
                - f_p_value (float): The p-value associated with the F-statistic.
        """
        # Ensure that X is a pandas DataFrame
        if not isinstance(self.dataholder.X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")
    
        # Apply the preprocessing transformer to the features
        X_trans = self.transformer.transform(self.dataholder.X_train)
        feature_names = [name.split("__")[-1] for name in self.transformer.get_feature_names_out()]
    
        n_samples, n_features = self.dataholder.X_train.shape
        residuals = self.dataholder.y_train - self.dataholder.predictions["train"]
    
        # Identify the model type (OLS, Lasso, ElasticNet)
        is_ols = isinstance(self.model, LinearRegression)
        is_reg = isinstance(self.model, (Lasso, ElasticNet))
    
        if not is_ols and not is_reg:
            raise ValueError("Unsupported model type. Must be LinearRegression, Lasso, or ElasticNet.")
    
        # Select the features based on model type (OLS or regularization)
        if is_ols:
            selected_idx = np.arange(n_features) # Select all features for OLS
        else:
            # Non-zero coefficients for Lasso/ElasticNet
            selected_idx = np.flatnonzero(self.model.coef_)
            if len(selected_idx) == 0:
                raise ValueError("No features selected (all coefficients are zero).")
    
        k = len(selected_idx)  # Number of selected features
        df_resid = n_samples - k - 1  # Degrees of freedom for residuals
        mse = np.sum(residuals ** 2) / df_resid  # Mean squared error (residuals)
    
        # Select relevant columns and add intercept for matrix calculations
        X_selected = X_trans[:, selected_idx]
        X_ = np.column_stack((np.ones(n_samples), X_selected))
        XtX_inv = np.linalg.inv(X_.T @ X_)
    
        # Calculate variance of coefficients and standard errors
        var_b = mse * XtX_inv
        se_b = np.sqrt(np.diag(var_b)) # Standard error of coefficients
    
        # Get the coefficients (including intercept)
        coefs = np.insert(self.model.coef_[selected_idx] if is_reg else self.model.coef_, 0, self.model.intercept_)
    
        # Calculate t-statistics and p-values
        t_stats = coefs / se_b
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df = df_resid))
    
        # Prepare feature names for the results
        selected_features = [feature_names[i] for i in selected_idx] if is_reg else feature_names
        full_feature_names = ['Intercept'] + list(selected_features)
    
        # Create a DataFrame with the coefficient statistics
        results_df = pd.DataFrame({
            'coefficient': coefs,
            'std_error': se_b,
            't_stat': t_stats,
            'p_value': p_values
        }, index = full_feature_names)
        results_df = results_df[results_df["p_value"] < 0.05].sort_values(["p_value", "coefficient"],
                                                                          ascending = [True, False])
    
        # Calculate the F-statistic for overall model fit
        ssr = np.sum((self.dataholder.predictions['train']
                      - np.mean(self.dataholder.y_train)) ** 2) # Sum of squared residuals
        msr = ssr / k # Mean squared regression
        f_stat = msr / mse # F-statistic
        f_p_value = 1 - stats.f.cdf(f_stat, k, df_resid) # p-value for the F-statistic
    
        return results_df, f_stat, f_p_value

    def evaluate_regression_model(self):
        """
        Generates visual diagnostics for LinearRegression, Lasso, or ElasticNet models.
        
        Parameters:
        - alpha: significance level for confidence intervals
        
        Visualizations:
        1. Residual vs Fitted Plot
        """
        coefficient_df, f_stat, f_p_value = self.regression_statistics()
        
        prompt = f"""
        Evaluate how effective the regression performed using the following information:
        
        - Features: {self.dataholder.X_train.columns}
        - Target: {self.dataholder.y_train.name}
        - Model type: {self.model_name}
        - Evaluate the coefficients of the equation using this data: {coefficient_df}
        - Model F-statistic: {f_stat}, F-p-value: {f_p_value}
        - Training and Testing scores: {self.dataholder.scores}

        The first time a technical term or abbreviation is used in the response, provide an explanation for that term.
        Once a term has been used, it does not need to be explained again.
        
        Provide an explanation of the top 5 most significant features.
        Follow the format: \"Feature Name (p-value, coefficient): Explanation\"
        A feature containing an underscore followed by a capital letter or number is a one-hot encoded feature
        and thus is a binary on/off flag. Do not mention unit increase in the explanation for this type of feature.
        
        Provide an evaluation of how well the model performs using hypothesis testing.
        Include this information in the section title:
            \"Hypothesis Testing (f-stat: {f_stat:.02f}, f-p-value: {f_p_value:.02f}):\"

        Provide an explanation of whether the model is overfitting or underfitting.
        Be sure to explain what overfitting or underfitting means.
        R^2 scores less than 0.7 means the model is underfitting.
        R^2 scores greater than 0.9 are excellent.
        If train R^2 is higher than test R^2 and train RSME is lower than test RSME, the model is overfitting.
        Explain what R^2 and RMSE measure.
        Include the RMSE and R^2 values in your explanation, as well as the scale of the target variable.
        Show the difference between train and test scores to justify your explanation.
        Include this information in the section title:
            \"Model Fit (train R^2: {self.dataholder.scores['train']['r2_score']:.02f},
                            train RSME: {self.dataholder.scores['train']['rsme']:.02f},
                            test R^2: {self.dataholder.scores['test']['r2_score']:.02f},
                            test RSME: {self.dataholder.scores['train']['rsme']:.02f}):\"

        For all referenced values, round all values to 2 decimal places.
        If the value is larger than 10,000 or less than 0.001, use scientific notation.
        Use commas for thousands separators.

        Make sure the title of each section is bolded.
        Make sure your explanations are understandable in layman's terms.
        Assume your audience has no prior knowledge about any technical terms.
        Answer in an active voice, not a passive voice.
        Summarize the overall feedback at the top of the response before addressing specific parts.
        Provide a brief reminder about the purpose of the model.
        
        Provide a rating using the format: \"Rating (X / 10)\" at the top of your response, after the summary.
        
        End your response with suggestions for model improvement.
        Do not recommend classification models or to change the scoring metrics.
        """
        judge_eval = self.judge.call_llama(prompt = prompt, save_history = False)

        display(Markdown(judge_eval))

        # 1. Residual vs Fitted Plot
        residuals = self.dataholder.y_train - self.dataholder.predictions['train']
        plt.figure(figsize = (10, 6))
        sns.residplot(x = self.dataholder.predictions['train'], y = residuals,
                      lowess = True, line_kws = {'color': 'red'})
        plt.title("Residual vs Fitted Plot", fontsize = 14)
        plt.xlabel("Predicted Values", fontsize = 12)
        plt.ylabel("Residuals (Actual - Predicted Values", fontsize = 12)
        plt.grid(True)
        plt.show()

        prompt = f"""
        Explain the residual plot in layman's terms:
        Predictions: {self.dataholder.predictions['train']}
        Residuals: {residuals}

        Do not rate the design of the plot, only the contents.
        """
        judge_plot_explanation = self.judge.call_llama(prompt = prompt, save_history = False)

        display(Markdown(judge_plot_explanation))

        return
