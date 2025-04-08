import numpy as np
import pandas as pd
from typing import Any, Dict, Union

class DataHolder():
    def __init__(self):
        # Define placeholder attributes
        self.csv_path = None
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.target_encoder = None
        self.regression_flag = None
        self.hyperparameter_tuning = None
        self.extended_models = None
        self.cv = None
        self.random_state = None
        self.best_model = None
        self.predictions = {'train': {}, 'test': {}}
        self.scores = {'train': {}, 'test': {}}

    def set_csv_path(self, csv_path: str) -> None:
        self.csv_path = csv_path
        return

    def set_modeling_parameters(self, hyperparameter_tuning: bool = False,
                                extended_models: bool = False, cv: int = 5, random_state: int = 42):
        self.hyperparameter_tuning = hyperparameter_tuning
        self.extended_models = extended_models
        self.cv = cv
        self.random_state = random_state
        return

    def get_modeling_parameters(self) -> Dict[str, Union[bool, int, None]]:
        return {
            'hyperparameter_tuning': self.hyperparameter_tuning,
            'extended_models': self.extended_models,
            'cv': self.cv,
            'random_state': self.random_state
        }
        
    def validate_modeling_parameters(self):
        kwargs = {key: value for key, value in self.get_modeling_parameters().items() if value is not None}
        self.set_modeling_parameters(**kwargs)
        return
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the internal pandas DataFrame.
    
        Returns:
            pd.DataFrame: The DataFrame stored within the object.
        """
        return self.df

    def get_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns the training and testing datasets.
    
        Returns:
            tuple: A tuple containing:
                - X_train (pd.DataFrame): Features for the training set.
                - X_test (pd.DataFrame): Features for the testing set.
                - y_train (pd.Series): Target values for the training set.
                - y_test (pd.Series): Target values for the testing set.
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data_encoders(self) -> Dict[str, Any]:
        """
        Retrieves the dictionary of data encoders used in the preprocessing pipeline.
    
        Returns:
            Dict[str, Any]: A dictionary mapping encoder names to their corresponding fitted encoder objects.
                            Includes the feature preprocessor and, if the task is classification, a label encoder.
        """
        # Initialize encoder dictionary with the main feature preprocessor
        encoders = {"preprocessor": self.preprocessor}
    
        # For classification tasks, include the label encoder as well
        if not self.regression_flag:
            encoders["target_encoder"] = self.target_encoder
    
        return encoders

    def set_machine_learning_task(self, regression_flag: bool) -> None:
        """
        Sets the machine learning task type.
    
        Args:
            regression_flag (bool): Whether the task is regression (1) or classification (0)
        """
        # Store the task type: 0 = classification, 1 = regression
        self.regression_flag = regression_flag
    
        return

    def get_best_model(self) -> Any:
        """
        Retrieves the best model selected after training and evaluation.
    
        Returns:
            Any: The best-performing model object based on cross-validation or direct evaluation.
        """
        return self.best_model

    def get_predictions(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the predictions made by the best model on the training and test sets.
    
        Returns:
            Dict[str, np.ndarray]: A dictionary containing predictions for "train" and "test" data.
        """
        return self.predictions

    def get_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the evaluation scores of the best model for both training and test sets.
    
        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing metric scores for "train" and "test" data.
        """
        return self.scores
