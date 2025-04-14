from .chatbot import *
from .dataholder import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import confusion_matrix
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
            regression_flag (bool): Type of task; 0 for classification, 1 for regression.
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
                You follow prompts to the best of your abilities.
            """,
            temperature = 0.3
        )
    
    def regression_statistics(self) -> Tuple[pd.DataFrame, float, float, float, float]:
        """
        Calculates coefficient statistics (t-statistics, p-values), the F-statistic, and information criterion
        for regression models like LinearRegression, Lasso, or ElasticNet.
    
        Returns:
            results_df (pd.DataFrame): A summary table of regression results with rows corresponding to
                model coefficients (including the intercept) and columns for:
                    - 'coefficient': Estimated coefficient values.
                    - 'std_error': Standard errors of the coefficients.
                    - 't_stat': t-statistics for testing whether each coefficient is zero.
                    - 'p_value': Two-tailed p-values corresponding to the t-statistics.
                    - '0.025': Lower bound of the 95% confidence interval.
                    - '0.975': Upper bound of the 95% confidence interval.
            f_stat (float): The F-statistic value testing the joint significance of the model coefficients.
            f_p_value (float): The p-value corresponding to the F-statistic.
            aic (float): Akaike Information Criterion, a model quality measure that penalizes model complexity.
            bic (float): Bayesian Information Criterion, similar to AIC but with a stronger penalty
                for the number of parameters.
        """
        # Ensure that X is a pandas DataFrame
        if not isinstance(self.dataholder.X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")
    
        # Apply the preprocessing transformer to the features
        X_trans = self.transformer.transform(self.dataholder.X_train)
        feature_names = [name.split("__")[-1] for name in self.transformer.get_feature_names_out()]
    
        n_samples, n_features = X_trans.shape
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

        # Calculate 95% confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_resid)
        ci_lower = coefs - t_crit * se_b
        ci_upper = coefs + t_crit * se_b
    
        # Prepare feature names for the results
        selected_features = [feature_names[i] for i in selected_idx] if is_reg else feature_names
        full_feature_names = ['Intercept'] + list(selected_features)
    
        # Create a DataFrame with the coefficient statistics
        results_df = pd.DataFrame({
            'coefficient': coefs,
            'std_error': se_b,
            't_stat': t_stats,
            'p_value': p_values,
            '0.025': ci_lower,
            '0.975': ci_upper
        }, index = full_feature_names)
        results_df = results_df[results_df["p_value"] < 0.05].sort_values(["p_value", "coefficient"],
                                                                          ascending = [True, False])
    
        # Calculate the F-statistic for overall model fit
        ssr = np.sum((self.dataholder.predictions['train']
                      - np.mean(self.dataholder.y_train)) ** 2) # Sum of squared residuals
        msr = ssr / k # Mean squared regression
        f_stat = msr / mse # F-statistic
        f_p_value = 1 - stats.f.cdf(f_stat, k, df_resid) # p-value for the F-statistic

        # Log-Likelihood assuming Gaussian residuals
        rss = np.sum(residuals ** 2)
        log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi * rss / n_samples) + 1)
        
        # Number of parameters (including intercept)
        n_params = k + 1
        
        # Akaike Information Criterion
        aic = -2 * log_likelihood + 2 * n_params
        
        # Bayesian Information Criterion
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
    
        return results_df, f_stat, f_p_value, aic, bic

    def plot_anova(self, coefficient_df: pd.DataFrame, f_stat: float,
                   f_p_value: float, aic: float, bic: float) -> str:
        """
        Generates a statsmodels-style formatted summary string for regression results.
    
        Args:
            coefficient_df (pd.DataFrame): A DataFrame containing regression output with the following columns:
                ['coefficient', 'std_error', 't_stat', 'p_value', '0.025', '0.975'], and variable names as the index.
            f_stat (float): The F-statistic for the overall model.
            f_p_value (float): The p-value associated with the F-statistic.
            aic (float): The Akaike Information Criterion for model selection.
            bic (float): The Bayesian Information Criterion for model selection.
    
        Returns:
            str: A formatted summary string showing model diagnostics and regression coefficients,
                 similar in style to the statsmodels OLS output.
        """
        # Extract model performance metrics
        r2_score = self.dataholder.scores["train"]["r2_score"]
        rmse = self.dataholder.scores["train"]["rmse"]
        n_samples = len(self.dataholder.y_train.index)
        p_features = len(coefficient_df.index)
        adjusted_r2 = 1 - ((1 - r2_score) * (n_samples - 1) / (n_samples - p_features - 1))
        df_resid = n_samples - p_features
        method = self.model_name.split("_")[0]
        title = f"{method.title()} Regression Results"
        dependent_variable = self.dataholder.y.name
    
        # Initialize summary content
        lines = []
        length = 110
        divider = '=' * length
        label_width = 35
        value_pad = 50  # Total width for label + value before the divider
    
        # Add title and header
        lines.append(f"{title.center(length)}")
        lines.append(divider)
    
        # Add general model statistics
        lines.append(f"{'Dependent Variable:':<{label_width}} {dependent_variable:<{value_pad - label_width}}| "
                     f"{'R-squared:':<{label_width}} {r2_score:>{length - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'Model:':<{label_width}} {method:<{value_pad - label_width}}| "
                     f"{'Adjusted R-squared:':<{label_width}} {adjusted_r2:>{length - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'No. Observations:':<{label_width}} {n_samples:<{value_pad - label_width}}| "
                     f"{'F-statistic:':<{label_width}} {f_stat:>{length - (2 * label_width) - value_pad}.3f}")
        lines.append(f"{'Df Residuals:':<{label_width}} {df_resid:<{value_pad - label_width}}| "
                     f"{'Prob (F-statistic):':<{label_width}} {f_p_value:>{length - (2 * label_width) - value_pad}.3g}")
        lines.append(f"{'Df Model:':<{label_width}} {p_features:<{value_pad - label_width}}| "
                     f"{'AIC:':<{label_width}} {aic:>{length - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'RMSE:':<{label_width}} {rmse:<{value_pad - label_width},.3f}| "
                     f"{'BIC:':<{label_width}} {bic:>{length - (2 * label_width) - value_pad},.3f}")
        lines.append(divider)
    
        # Add coefficient table header
        header = f"{'':<{label_width}} {'coef':^10} {'std err':^12} {'t':^10} {'P>|t|':^10} {'0.025':^12} {'0.975':^12}"
        lines.append(header)
        lines.append("-" * length)
    
        # Add rows for each coefficient
        for idx, row in coefficient_df.iterrows():
            lines.append(f"{idx:<{label_width}} "
                         f"{row['coefficient']:^10,.2g} "
                         f"{row['std_error']:^12,.2g} "
                         f"{row['t_stat']:^10.2f} "
                         f"{row['p_value']:^10.2g} "
                         f"{row['0.025']:^12,.2g} "
                         f"{row['0.975']:^12,.2g}")
    
        # Combine all lines into a single string
        return "\n".join(lines)

    def plot_residuals(self, residuals: np.array(float)) -> plt.Figure:
        """
        Creates a residual plot to visualize the difference between actual and predicted values,
        including highlighting of outliers based on standard deviation thresholds.
    
        Args:
            residuals (np.array(float)): The residuals (actual - predicted values) from the model.
    
        Returns:
            plt.Figure: A matplotlib Figure object containing the residual plot.
        """
    
        # Convert predictions and residuals to NumPy arrays for processing
        preds = np.array(self.dataholder.predictions['test'])
    
        # Calculate threshold for outlier detection (e.g., 3 standard deviations)
        std_resid = np.std(residuals)
        mean_resid = np.mean(residuals)
        threshold = 3 * std_resid
    
        # Identify outliers (points with residuals far from the mean)
        outlier_mask = np.abs(residuals - mean_resid) > threshold
        non_outlier_mask = ~outlier_mask
    
        # Create the figure
        fig = plt.figure(figsize = (10, 6))
    
        # Plot non-outliers with seaborn's residplot
        sns.residplot(
            x = preds[non_outlier_mask],
            y = residuals[non_outlier_mask],
            lowess = True,
            line_kws = {'color': 'red'},
            scatter_kws = {'alpha': 0.6, 'label': 'Non-Outliers'}
        )
    
        # Plot outliers as orange dots
        plt.scatter(
            preds[outlier_mask],
            residuals[outlier_mask],
            color = 'orange',
            edgecolor = 'black',
            label = 'Outliers',
            zorder = 3
        )
    
        # Add plot title and axis labels
        plt.title("Residual vs Fitted Plot with Outlier Detection", fontsize = 14)
        plt.xlabel("Predicted Values", fontsize = 12)
        plt.ylabel("Residuals (Actual - Predicted), Errors", fontsize = 12)
        plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 1)
    
        # Add legend and grid
        plt.legend()
        plt.grid(True)
    
        return fig

    def heatmap_confusion_matrix(self, is_test: bool, ax: plt.Axes) -> None:
        """
        Generates and plots a heatmap of the confusion matrix for classification results 
        using seaborn, along with classification metrics such as accuracy, precision, recall, and F1 score.
    
        Args:
            is_test (bool): Indicates whether to use the test set (True) or training set (False).
            ax (plt.Axes): Matplotlib Axes object on which the heatmap will be plotted.
        """
        
        # Select the appropriate true and predicted labels based on the mode (train/test)
        if is_test:
            y = self.dataholder.y_test
            y_pred = self.dataholder.predictions["test"]
            metrics = self.dataholder.scores["test"]
        else:
            y = self.dataholder.y_train
            y_pred = self.dataholder.predictions["train"]
            metrics = self.dataholder.scores["train"]
        
        # Compute the confusion matrix
        conf = confusion_matrix(y, y_pred)
        matrix_values = np.asarray([f'{value:0d}' for value in conf.flatten()]).reshape(*conf.shape)
        
        # Create class labels using the fitted target encoder
        # labels = self.dataholder.target_encoder.classes_.reshape(*conf.shape)
    
        # Plot the confusion matrix as a heatmap
        sns.heatmap(conf, annot = matrix_values, fmt = '', annot_kws = {'fontsize': 14},
                    cbar = False, ax = ax, cmap = 'Blues')
    
        # Generate a nicely formatted model name for the title
        model_name = ' '.join([substr.title() for substr in self.model_name.split('_')])
        model_name += ' Test' if is_test else ' Train'
        
        # Set plot title and axis labels
        ax.set_title(f'{model_name} Confusion Matrix', fontsize = 20)
        ax.set_xlabel('Predicted', fontsize = 16)
        ax.set_ylabel('Observed', fontsize = 16)
        ax.tick_params(axis = "both", labelsize = 14)
    
        # Extract and format key classification metrics
        accuracy = metrics.get("accuracy", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)
    
        # Create a formatted string of classification metrics
        classification_labels = '\n\n'.join([
            '\n'.join(metric) for metric in zip(
                ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1'],
                [f'{accuracy:.2f}', f'{precision:.2f}', f'{recall:.2f}', f'{f1:.2f}']
            )
        ])
    
        # Define style for the metrics text box
        props = dict(boxstyle = 'round', facecolor = 'grey', alpha = 0.01)
    
        # Display the classification metrics beside the heatmap
        ax.text(1.15, 0.5, classification_labels, fontsize = 14, transform = ax.transAxes,
                bbox = props, ha = 'center', va = 'center')
    
        return

    def plot_confusion_matrices(self) -> plt.Axes:
        """
        Generates side-by-side confusion matrix heatmaps for both training and test data.
    
        Returns:
            plt.Axes: A matplotlib Axes array containing the two confusion matrix subplots.
        """
    
        # Create a figure with two subplots: one for training, one for testing
        fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    
        # Loop through both training (flag = 0) and testing (flag = 1)
        for flag in [0, 1]:
            # Generate the confusion matrix heatmap on the corresponding subplot
            self.heatmap_confusion_matrix(is_test = bool(flag), ax = ax[flag])
    
        # Adjust subplot spacing
        plt.tight_layout()
    
        return ax