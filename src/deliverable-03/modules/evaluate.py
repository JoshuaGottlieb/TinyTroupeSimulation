# Absolute imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Third-party imports
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import shap

# Local imports
from .chatbot import LlamaBot
from .dataholder import DataHolder

# Typing imports
from typing import Tuple, List

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
        self.model_name = f"{' '.join([s.title() for s in self.model_name.split('_')])}"
        self.transformer = self.dataholder.best_model.named_steps['preprocess']
        
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
        t_crit = stats.t.ppf(1 - alpha / 2, df = df_resid)
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
                   f_p_value: float, aic: float, bic: float, max_chars: int = 100,
                   rows_per_page: int = 25) -> str:
        """
        Generates a statsmodels-style formatted summary string for regression results,
        constrained by line width and paginated for long coefficient tables.
    
        Args:
            coefficient_df (pd.DataFrame): DataFrame containing regression coefficients and statistics.
            f_stat (float): F-statistic value for the overall model.
            f_p_value (float): P-value associated with the F-statistic.
            aic (float): Akaike Information Criterion value for model comparison.
            bic (float): Bayesian Information Criterion value for model comparison.
            max_chars (int): Maximum number of characters per line.
            rows_per_page (int): Max number of coefficient rows before splitting across pages.
    
        Returns:
            str: A formatted multiline string summarizing regression model results.
        """
        # Retrieve key regression diagnostics
        r2_score = self.dataholder.regression_scores["train"]["r2_score"]
        rmse = self.dataholder.regression_scores["train"]["rmse"]
        n_samples = len(self.dataholder.y_train.index)
        p_features = len(coefficient_df.index)
        adjusted_r2 = 1 - ((1 - r2_score) * (n_samples - 1) / (n_samples - p_features - 1))
        df_resid = n_samples - p_features
        method = self.model_name.split("Regression")[0].strip().lower()
        title = f"{method.title()} Regression Results"
        dependent_variable = self.dataholder.y_train.name
    
        # Set up layout constants
        lines = []
        divider = "=" * max_chars
        header_width = max_chars
        label_width = 28
        value_pad = 48
    
        # Header Section
        lines.append(f"{title.center(header_width)}")
        lines.append(divider)
        lines.append(f"{'Dependent Variable:':<{label_width}} {dependent_variable:<{value_pad - label_width}}| "
                     f"{'R-squared:':<{label_width}} {r2_score:>{max_chars - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'Model:':<{label_width}} {method:<{value_pad - label_width}}| "
                     f"{'Adjusted R-squared:':<{label_width}} {adjusted_r2:>{max_chars - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'No. Observations:':<{label_width}} {n_samples:<{value_pad - label_width}}| "
                     f"{'F-statistic:':<{label_width}} {f_stat:>{max_chars - (2 * label_width) - value_pad}.3f}")
        lines.append(f"{'Df Residuals:':<{label_width}} {df_resid:<{value_pad - label_width}}| "
                     f"{'Prob (F-statistic):':<{label_width}} {f_p_value:>{max_chars - (2 * label_width) - value_pad}.3g}")
        lines.append(f"{'Df Model:':<{label_width}} {p_features:<{value_pad - label_width}}| "
                     f"{'AIC:':<{label_width}} {aic:>{max_chars - (2 * label_width) - value_pad},.3f}")
        lines.append(f"{'RMSE:':<{label_width}} {rmse:<{value_pad - label_width},.3f}| "
                     f"{'BIC:':<{label_width}} {bic:>{max_chars - (2 * label_width) - value_pad},.3f}")
        lines.append(divider)
    
        # Coefficient table headers
        def add_coef_header():
            lines.append(f"{'':<{label_width}} {'coef':^10} {'std err':^12} {'t':^10} "
                         f"{'P>|t|':^10} {'0.025':^12} {'0.975':^12}")
            lines.append("-" * max_chars)
    
        add_coef_header()
    
        # Add rows with page breaks if needed
        for i, (idx, row) in enumerate(coefficient_df.iterrows()):
            if i > 0 and i % rows_per_page == 0:
                lines.append("\f")  # Form feed character to signal page break
                add_coef_header()
            lines.append(f"{idx:<{label_width}} "
                         f"{row['coefficient']:^10,.2g} "
                         f"{row['std_error']:^12,.2g} "
                         f"{row['t_stat']:^10.2f} "
                         f"{row['p_value']:^10.2g} "
                         f"{row['0.025']:^12,.2g} "
                         f"{row['0.975']:^12,.2g}")
    
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
        # preds = np.array(self.dataholder.predictions['test'])
        preds = self.dataholder.predictions['test']
    
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
            ax (plt.Axes): Matplotlib ax object on which the heatmap will be plotted.
        """
        
        # Select the appropriate true and predicted labels based on the mode (train/test)
        if is_test:
            y = self.dataholder.y_test
            y_pred = self.dataholder.predictions["test"]
            metrics = self.dataholder.classification_scores["test"]
        else:
            y = self.dataholder.y_train
            y_pred = self.dataholder.predictions["train"]
            metrics = self.dataholder.classification_scores["train"]
        
        # Compute the confusion matrix
        conf = confusion_matrix(y, y_pred)
        matrix_values = np.asarray([f'{value:0d}' for value in conf.flatten()]).reshape(*conf.shape)
    
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

    def plot_confusion_matrices(self) -> plt.Figure:
        """
        Generates side-by-side confusion matrix heatmaps for both training and test data.
    
        Returns:
            plt.Figure: A matplotlib Figure containing the confusion matrices.
        """
    
        # Create a figure with two subplots: one for training, one for testing
        fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    
        # Loop through both training (flag = 0) and testing (flag = 1)
        for flag in [0, 1]:
            # Generate the confusion matrix heatmap on the corresponding subplot
            self.heatmap_confusion_matrix(is_test = bool(flag), ax = ax[flag])
    
        # Adjust subplot spacing
        plt.tight_layout()
    
        return fig

    def plot_roc_curve(self) -> Tuple[plt.Figure, float]:
        """
        Plots the Receiver Operating Characteristic (ROC) curve for a trained classification model.
        
        This method generates either a binary or macro-averaged multiclass ROC curve, depending on the
        number of unique target classes in the training set. The ROC curve visualizes the model's
        ability to distinguish between classes at various threshold settings. It also includes
        annotations for the area under the curve (AUC), a random classifier, and a perfect classifier.
        
        Returns:
            Tuple[plt.Figure, float]:
                - A Matplotlib figure object containing the ROC curve plot.
                - A float representing the AUC (Area Under the Curve) score.
        """
        # Determine the number of classes in the training target
        n_classes = len(self.dataholder.y_train.unique())
    
        # Get the model's predicted probabilities on the test set
        y_score = self.dataholder.best_model.predict_proba(self.dataholder.X_test)
    
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize = (16, 8))
    
        if n_classes == 2:
            # Binary classification ROC curve
            fpr, tpr, _ = roc_curve(self.dataholder.y_test, y_score[:, 1])
            roc_auc = roc_auc_score(self.dataholder.y_test, y_score[:, 1])
    
            # Plot ROC curve for binary classification
            ax.plot(
                fpr, tpr,
                label = f"{self.model_name} ROC Curve (AUC = {roc_auc:.2f})",
                color = "cornflowerblue", linewidth = 4
            )
            ax.set_title(f"Receiver Operating Characteristic for {self.model_name}", fontsize = 20)
    
        else:
            # Multiclass classification: compute macro-average ROC
            label_binarizer = LabelBinarizer().fit(self.dataholder.y_train)
            y_onehot_test = label_binarizer.transform(self.dataholder.y_test)
    
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
    
            # Compute ROC curve and interpolate for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
                mean_tpr += np.interp(fpr_grid, fpr, tpr)
    
            # Average the TPRs and compute overall AUC
            mean_tpr /= n_classes
            roc_auc = roc_auc_score(
                self.dataholder.y_test, y_score,
                multi_class = "ovr", average = "macro"
            )
    
            # Plot the macro-average ROC curve
            ax.plot(
                fpr_grid, mean_tpr,
                label = f"{self.model_name} Macro-Average ROC Curve (AUC = {roc_auc:.2f})",
                color = "cornflowerblue", linewidth = 4
            )
            ax.set_title(f"Macro-Average Receiver Operating Characteristic for {self.model_name}", fontsize = 20)
    
        # Plot reference lines for perfect and random classifiers
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axline((0, 0), slope = 1, color = "black", linestyle = "--", label = "Random Chance Classifier (AUC = 0.5)")
        ax.axhline(ax.get_ylim()[1] * 0.995, color = "green", linestyle = "--", label = "Perfect Classifier (AUC = 1)")
    
        # Add labels and formatting
        ax.set_xlabel("False Positive Rate", fontsize = 16)
        ax.set_ylabel("True Positive Rate", fontsize = 16)
        ax.tick_params(axis = "both", labelsize = 14)
        ax.legend(fontsize = 12, loc = 4)
    
        return fig, roc_auc

    def get_shap_scores(
        self, explainer: shap.Explainer, X: pd.DataFrame,
        feature_names: List[str], top_n: int = 10
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Calculates SHAP importance scores and returns the top N most impactful features.
    
        This method computes both the average and maximum absolute SHAP values for each feature
        across all predictions, and returns the top N features ranked by those scores.
    
        Args:
            explainer (shap.Explainer): A fitted SHAP explainer object (e.g., TreeExplainer, KernelExplainer).
            X (pd.DataFrame): The dataset to compute SHAP values for.
            feature_names (List[str]): A list of feature names matching the columns in X.
            top_n (int, optional): Number of top features to return. Defaults to 10.
    
        Returns:
            Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
                - top_avg_scores: Top N features ranked by average absolute SHAP value.
                - top_max_scores: Top N features ranked by maximum absolute SHAP value.
        """
        # Compute SHAP values for the input data
        shap_values = explainer(X)
    
        # Get absolute SHAP values to assess feature importance (handles both single-output and multi-output cases)
        values_array = np.abs(shap_values.values if hasattr(shap_values, 'values') else shap_values)
    
        # Calculate average and maximum SHAP values across all samples
        avg_scores = np.round(values_array.mean(axis = 0), decimals = 2)
        max_scores = np.round(values_array.max(axis = 0), decimals = 2)
    
        # Pair each score with the corresponding feature name
        avg_pairs = list(zip(feature_names, avg_scores))
        max_pairs = list(zip(feature_names, max_scores))
    
        # Sort the scores in descending order and extract the top N features
        top_avg_scores = sorted(avg_pairs, key = lambda x: x[1], reverse = True)[:top_n]
        top_max_scores = sorted(max_pairs, key = lambda x: x[1], reverse = True)[:top_n]
    
        return top_avg_scores, top_max_scores
        
    def shap_summary_plot(self) -> Tuple[plt.Figure, List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Generates SHAP summary bar plots to visualize feature importance for a trained model.
        
        This method:
        - Transforms the test set using the model's preprocessing pipeline.
        - Computes SHAP values to quantify the contribution of each feature to model predictions.
        - Extracts and ranks the top features by average and maximum absolute SHAP values.
        - Plots two horizontal bar charts:
            - Left: Features sorted by average absolute SHAP value (overall importance).
            - Right: Features sorted by maximum absolute SHAP value (peak importance in any sample).
        
        Returns:
            Tuple[plt.Figure, List[Tuple[str, float]], List[Tuple[str, float]]]:
                - A Matplotlib figure containing two SHAP bar plots side by side.
                - A list of tuples with the top features by average absolute SHAP values.
                - A list of tuples with the top features by maximum absolute SHAP values.
        """
        # Transform test data using the model's preprocessing pipeline
        X_test = self.transformer.transform(self.dataholder.X_test)
    
        # Clean up feature names for readability from transformer output
        feature_names = [
            " ".join((name.split("__")[-1].split('_')))
            for name in self.transformer.get_feature_names_out()
        ]
    
        # Create a SHAP explainer using the model’s prediction method
        explainer = shap.Explainer(self.model.predict, X_test, feature_names = feature_names)
    
        # Get SHAP scores: average and max importance per feature
        top_avg_scores, top_max_scores = self.get_shap_scores(explainer, X_test, feature_names)
    
        # Separate names and values for plotting
        avg_features, avg_values = zip(*top_avg_scores)
        max_features, max_values = zip(*top_max_scores)
    
        # Set up a side-by-side horizontal bar chart layout
        fig, ax = plt.subplots(1, 2, figsize = (16, 8))
        fig.suptitle('Top SHAP Feature Importances', fontsize = 20)
    
        # Plot average SHAP values
        ax[0].barh(avg_features[::-1], avg_values[::-1], color = 'skyblue')
        ax[0].set_title("Average Importance per Feature", fontsize = 16)
        ax[0].set_xlabel("|mean SHAP Value|", fontsize = 14)
        ax[0].tick_params(axis = "both", labelsize = 14)
    
        # Plot maximum SHAP values
        ax[1].barh(max_features[::-1], max_values[::-1], color = 'salmon')
        ax[1].set_title("Maximum Importance per Feature", fontsize = 16)
        ax[1].set_xlabel("|max SHAP Value|", fontsize = 14)
        ax[1].tick_params(axis = "both", labelsize = 14)
    
        # Adjust layout for better spacing
        plt.tight_layout(rect = [0, 0, 1, 0.95])
    
        return fig, top_avg_scores, top_max_scores