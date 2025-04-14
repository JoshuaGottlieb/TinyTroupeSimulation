from .chatbot import *
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
from typing import Any, Dict, Union, List

class DataHolder():
    def __init__(self):
        pass

    def get(self, name: str, value: Any = None) -> Any:
        return getattr(self, name, value)

    def set(self, name: str, value: Any) -> None:
        setattr(self, name, value)
        return

    def validate_parameter(self, parameter: str, correct_type: str) -> bool:
        value = self.get(parameter)

        return type(value) == correct_type
    
    def validate_parameters(self, parameters: Dict[str, str]) -> list:
        missing_parameters = []
        
        for parameter, type_ in parameters.items():
            if not self.validate_parameter(parameter, type_):
                missing_parameters.append(parameter)

        return missing_parameters

    def get_parameters(self, parameters: list[str]) -> Dict[str, Any]:
        return {parameter: self.get(parameter) for parameter in parameters}

    def delete_attribute(self, name: str) -> None:
        delattr(self, name)

        return

    def delete_all_attributes(self, exceptions: List[str] = []) -> None:
        attributes = list(self.__dict__.keys())
        for attribute in attributes:
            if attribute not in exceptions:
                self.delete_attribute(attribute)

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
            'hyperparameter_tuning': self.get("hyperparameter_tuning"),
            'extended_models': self.get("extended_models"),
            'cv': self.get("cv", 5),
            'random_state': self.get("random_state", 42)
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

    def set_machine_learning_task(self, regression_flag: bool) -> None:
        """
        Sets the machine learning task type.
    
        Args:
            regression_flag (bool): Whether the task is regression (1) or classification (0)
        """
        # Store the task type: 0 = classification, 1 = regression
        self.regression_flag = regression_flag
    
        return

    def load_dataframe(self) -> bool:
        try:
            self.df = pd.read_csv(self.csv_path)
            return True
        except:
            return False
    
    def normalize_column_names(self) -> bool:
        """
        Normalizes the column names of the internal DataFrame to snake_case format.
    
        Returns:
            None
        """
        # Instantiate a chatbot to process data names.
        helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)
        
        # Use a large language model to intelligently convert column names
        prompt = f"""
        Given the following list of strings, convert the list of strings to snake case format.
        If a column is already in snake case format, simply convert to lowercase.
        Columns: {self.df.columns}
        Do not give an explanation, only return a list of strings.
        """
        
        response = helper.call_llama_eval(prompt)

        # If the response type is wrong, may fail, so wrap in a try-except block
        try:
            self.df.columns = response
            return True
        except:
            return False

    def convert_column_types(self) -> bool:
        """
        Infers and converts column data types in the DataFrame using LlamaBot assistance.
    
        Returns:
            None
    
        Raises:
            Prints an error message if type conversion fails.
        """
        # Instantiate a chatbot to process data names.
        helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)
        
        # Construct a prompt to get suggested data types from the language model
        prompt = f"""
        Determine which data types should be used for each column of the following Pandas dataframe:
        
        DataFrame: {self.df.head(10)}
        
        Do not give an explanation.
        Do not assign a data type of str.
        If a column has empty strings but is otherwise numeric, assign an appropriate numeric data type of int or float.
        Convert datetime columns as necessary by assigning a datatype of exactly datetime64[ns].
        For other columns containing text data, assign a data type of exactly object.
        Provide only a dictionary with keys as column names and values as the assigned pandas data types.
        """
        # Call LlamaBot to get a column-type mapping
        response = helper.call_llama_eval(prompt)
        
        try:
            for column in response.keys():
                unique_values = self.df[column].unique()

                # Drop columns with high uniqueness, as they are not discriminatory
                # Likely includes information such as IDs, UUIDs, full addresses, etc.
                # Drop columns with all unique values (likely IDs or UUIDs)
                if len(unique_values) >= (0.5 * len(self.df.index)):
                    self.df = self.df.drop(column, axis = 1)
    
                # Handle binary columns by mapping to boolean integers
                elif len(unique_values) == 2:
                    prompt = f"""
                    Give a mapping to convert the values to a boolean variable.
                    values: {unique_values}
                    
                    Use the original value as the key, do not convert the original value's data type.
                    Use the appropriate boolean value cast as an integer for the value.
                    Return a dictionary.
                    """
                    mapping_dict = helper.call_llama_eval(prompt)
                    self.df[column] = self.df[column].map(mapping_dict).astype(np.int64)
    
                # Handle all other columns based on inferred type
                else:
                    numeric = is_numeric_dtype(response[column])
                    if numeric:
                        # Attempt to convert to a numeric type (int or float)
                        self.df[column] = pd.to_numeric(self.df[column], errors = "coerce")
                    else:
                        # Convert to the inferred non-numeric type
                        self.df[column] = self.df[column].astype(response[column])
    
            print("The current version cannot support date or time columns. These will be dropped automatically.")
    
            # Drop datetime columns (unsupported)
            self.df = self.df.select_dtypes(include = ["number", "object"])

            return True
    
        except Exception as e:
            print("Unable to determine data types for columns. This may result in unexpected behavior for preprocessing.")
            print(e)

            return False
    
    def split_features(self, y: str) -> bool:
        """
        Splits the DataFrame into features (X) and target (y).
    
        Args:
            y (str): The name of the target column.
        """
        try:
            self.df = self.df.dropna(subset = [y])
            self.X = self.df.drop(y, axis = 1)
            self.y = self.df[y]
            return True
        except:
            return False