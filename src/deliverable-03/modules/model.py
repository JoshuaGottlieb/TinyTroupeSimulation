# Absolute imports
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd

# Local imports
from .chatbot import *
from .dataholder import *

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category = ConvergenceWarning)

class AutomaticModeler:
    def __init__(self, dataholder: DataHolder):
        """
        Initializes the model evaluation pipeline with training and testing datasets,
        task configuration, and optional modeling settings.
    
        Args:
            X_train (pd.DataFrame): Feature matrix for training.
            X_test (pd.DataFrame): Feature matrix for testing.
            y_train (pd.Series): Target vector for training.
            y_test (pd.Series): Target vector for testing.
            regression_flag (bool): Task type flag — True for regression, False for classification.
            encoders (Dict[str, Any]): Dictionary of fitted data encoders used for preprocessing.
            hyperparameter_tuning (bool, optional): If True, enables hyperparameter tuning. Default is False.
            extended_models (bool, optional): If True, includes additional model types during evaluation. Default is False.
            cv (int, optional): Number of cross-validation folds. Default is 5.
            random_state (int, optional): Seed used for reproducibility. Default is 42.
        """
        self.dataholder = dataholder
        self.dataholder.validate_modeling_parameters()
        
        # Initialize models and performance metrics dictionary
        self.determine_models_and_metrics()
    
        # Placeholder for the best-performing model and results
    
    def determine_models_and_metrics(self) -> None:
        """
        Determines the appropriate machine learning models and evaluation metrics
        based on the task type (classification or regression).
        """
        self.models = []
        if self.dataholder.regression_flag:
            # Task is regression
            # Add base regression model
            self.models.append(Pipeline(steps = [("linear_regression", LinearRegression())]))
    
            # Optionally add more complex regression models
            if self.dataholder.extended_models:
                self.models.append(Pipeline(steps = [("elastic_net", ElasticNet())]))
                self.models.append(Pipeline(steps = [("polynomial_regression", Lasso())]))
    
            # Initialize regression metrics
            self.metrics = {
                "rmse": root_mean_squared_error,
                "r2_score": r2_score
            }
        else:
            # Task is classification
            # Add base classification model
            self.models.append(Pipeline(steps = [("logistic_regression", LogisticRegression())]))
    
            # Optionally add more complex classification models
            if self.dataholder.extended_models:
                self.models.append(Pipeline(steps = [
                    (
                        "decision_tree",
                        DecisionTreeClassifier(random_state=self.dataholder.random_state)
                    )
                ]))
                self.models.append(Pipeline(steps = [
                    (
                        "random_forest",
                        RandomForestClassifier(random_state=self.dataholder.random_state)
                    )
                ]))
    
            # Initialize classification metrics
            self.metrics = {
                "accuracy": accuracy_score,
                "recall": recall_score,
                "precision": precision_score,
                "f1_score": f1_score
                # "roc_auc": roc_auc_score
            }
    
        return

    def build_pipeline(self, model: Pipeline) -> Pipeline:
        """
        Constructs a complete machine learning pipeline by combining preprocessing steps with the provided model.
    
        If the model is a polynomial regression, this method dynamically replaces the standard scaler with
        polynomial feature expansion for numeric columns with more than two unique values.
    
        Args:
            model (Pipeline): A scikit-learn pipeline that includes the model step.
    
        Returns:
            Pipeline: A full pipeline with preprocessing and modeling steps.
        """
        # Get the name of the model from the first step of the passed-in pipeline
        name = model.steps[0][0]
    
        # Retrieve the base preprocessing encoder from the stored encoders
        preprocessing_encoder = deepcopy(self.dataholder.preprocessor)
    
        # If the selected model is polynomial regression, modify the preprocessing to use polynomial features
        if name == "polynomial_regression":
            # Identify numeric columns with more than two unique values (skip binary)
            numeric_columns = [
                column for column in self.dataholder.X_train.select_dtypes('number').columns
                    if len(self.dataholder.X_train[column].unique()) > 2
            ]
    
            # Replace the numerical preprocessing transformer with PolynomialFeatures
            preprocessing_encoder.transformers[-1] = (
                "poly",
                PolynomialFeatures(include_bias = False),
                numeric_columns
            )

            # Define new feature names
            number_expanded_ohe_features = np.array(
                [
                    len(self.dataholder.X_train[column].unique()) - 1
                    for column in self.dataholder.X_train if column not in numeric_columns
                ]
            ).sum()
            polynomial_slice = slice(number_expanded_ohe_features, None)
            scaler = ColumnTransformer(
                transformers = [
                    (
                        "ssc",
                        StandardScaler(),
                        polynomial_slice
                    )
                ], remainder = "passthrough"
            )

            pipeline = Pipeline(steps = [("preprocess", preprocessing_encoder), ("scaler", scaler), ("estimator", model)])

        else:
            # Combine the preprocessing and model steps into a unified pipeline
            pipeline = Pipeline(steps = [("preprocess", preprocessing_encoder), ("estimator", model)])
        
    
        return pipeline
    
    def get_gridsearch(self, pipeline: Pipeline) -> GridSearchCV:
        """
        Returns a GridSearchCV object for hyperparameter tuning based on the model in the given pipeline.
    
        Args:
            pipeline (Pipeline): A scikit-learn pipeline with a named model step.
    
        Returns:
            GridSearchCV: A GridSearchCV object configured with hyperparameter ranges.
        """
        # Get the name of the model from the first step of the pipeline
        name = list(pipeline.named_steps["estimator"].named_steps.keys())[0]
    
        # Define hyperparameter search space for different model types
        hyperparameters = {
            "linear_regression": {},
            "elastic_net": {
                f"estimator__{name}__alpha": np.logspace(-3, 1, 4),
                f"estimator__{name}__l1_ratio": [0.1, 0.5, 0.9]
            },
            "polynomial_regression": {
                "preprocess__poly__degree": [2, 3],
                f"estimator__{name}__alpha": np.logspace(4, 8, 4)
            },
            "logistic_regression": {
                f"estimator__{name}__C": np.logspace(-3, 3, 6)
            },
            "decision_tree": {
                f"estimator__{name}__max_depth": [None, 5, 10],
                f"estimator__{name}__max_features": [None, "sqrt", "log2"]
            },
            "random_forest": {
                f"estimator__{name}__max_depth": [None, 5, 10]
            }
        }

        # Use negative RSME scoring for regression, ROC AUC scoring for classification
        scoring = "neg_root_mean_squared_error" if self.dataholder.regression_flag else "accuracy"
        
        # Create the GridSearchCV object using the appropriate hyperparameters
        grid = GridSearchCV(
            estimator = pipeline,
            param_grid = hyperparameters[name],
            cv = self.dataholder.cv,
            scoring = scoring
        )
    
        return grid

    def cross_validate_prefitted_models(self) -> int:
        """
        Performs k-fold cross-validation on prefitted models to determine the best model based on a scoring metric.
    
        The function uses the provided number of cross-validation splits (self.dataholder.cv) and evaluates each model
        using the primary metric defined in self.metrics. The best model is selected based on the average
        performance across all folds.
    
        Returns:
            int: The index of the best-performing model based on the cross-validation score.
        """
        # Initialize KFold cross-validation
        kf = KFold(n_splits = self.dataholder.cv, shuffle = True, random_state = self.dataholder.random_state)
    
        # Initialize a dictionary to store results for each model
        results = {f"model_{i + 1}": [] for i in range(len(self.models))}
    
        # Use the first metric in the self.metrics dictionary as the scoring function
        scoring = list(self.metrics.values())[0]
    
        # Set polarity: negative for regression (minimize), positive for classification (maximize)
        polarity = -1 if self.dataholder.regression_flag else 1
    
        # Perform cross-validation over each fold
        for fold_index, val_index in kf.split(self.dataholder.X_train):
            # Split data into training and validation sets for this fold
            X_fold, X_val = self.dataholder.X_train.iloc[fold_index], self.dataholder.X_train.iloc[val_index]
            y_fold, y_val = self.dataholder.y_train.iloc[fold_index], self.dataholder.y_train.iloc[val_index]
    
            # Evaluate each model on the validation set
            for i, model in enumerate(self.models):
                y_pred = model.predict(X_val)
                if self.dataholder.regression_flag:
                    score = scoring(y_val, y_pred)
                else:
                    score = scoring(y_val, y_pred)
    
                # Store polarity-adjusted score
                results[f"model_{i + 1}"].append(polarity * score)
    
        # Calculate the average score for each model across all folds
        average_scores = [np.array(cv_scores).mean() for cv_scores in results.values()]
    
        # Select the model with the highest average score (adjusted for polarity)
        best_model_idx = np.argmax(average_scores)
    
        return best_model_idx

    def fit_and_score_model(self) -> bool:
        """
        Trains, tunes, and evaluates machine learning models using training and test data.
    
        This method builds pipelines for each model, optionally performs hyperparameter tuning using
        GridSearchCV, fits the models to the training data, and identifies the best model using
        cross-validation if multiple models are present.
    
        After fitting, the best model is used to make predictions on both the training and test sets.
        Each set of predictions is then scored using the predefined metrics (e.g., accuracy, RMSE),
        and the results are stored in the `self.dataholder.scores` dictionary.
        """
        try:
            self.dataholder.predictions = {"train": [], "test": []}
            if self.dataholder.regression_flag:
                self.dataholder.regression_scores = {"train": {}, "test": {}}
            else:
                self.dataholder.classification_scores = {"train": {}, "test": {}}
            
            for i, model in enumerate(self.models):
                # Create a pipeline that includes preprocessing and the model
                pipeline = self.build_pipeline(model)
        
                if self.dataholder.hyperparameter_tuning:
                    # If tuning is enabled, perform grid search with cross-validation
                    grid = self.get_gridsearch(pipeline)
                    grid.fit(self.dataholder.X_train, self.dataholder.y_train)
        
                    # Update model with the best estimator found during tuning
                    self.models[i] = grid.best_estimator_
                else:
                    # Otherwise, fit the pipeline directly
                    pipeline.fit(self.dataholder.X_train, self.dataholder.y_train)
                    self.models[i] = pipeline
        
            # If multiple models are available, use cross-validation to select the best one
            if len(self.models) > 1:
                self.dataholder.best_model = self.models[self.cross_validate_prefitted_models()]
            else:
                self.dataholder.best_model = self.models[0]
        
            # Make predictions using the selected best model
            self.dataholder.predictions["train"] = self.dataholder.best_model.predict(self.dataholder.X_train)
            self.dataholder.predictions["test"] = self.dataholder.best_model.predict(self.dataholder.X_test)
        
            # Evaluate and store scores for each metric on both train and test sets
            for metric, scorer in self.metrics.items():
                if self.dataholder.regression_flag: # Regression
                    self.dataholder.regression_scores["train"][metric] = scorer(
                        self.dataholder.y_train, self.dataholder.predictions["train"]
                    )
                    self.dataholder.regression_scores["test"][metric] = scorer(
                        self.dataholder.y_test, self.dataholder.predictions["test"]
                    )
                else: # Classification
                    kwargs = {}
                    if metric not in ["accuracy"]:
                        kwargs["average"] = "macro"
                    self.dataholder.classification_scores["train"][metric] = scorer(
                        self.dataholder.y_train, self.dataholder.predictions["train"], **kwargs
                    )
                    self.dataholder.classification_scores["test"][metric] = scorer(
                        self.dataholder.y_test, self.dataholder.predictions["test"], **kwargs
                    )
    
            return True
        except Exception as e:
            print(e)
            return False