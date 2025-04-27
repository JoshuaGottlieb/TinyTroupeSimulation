# Absolute imports
import numpy as np
import pandas as pd

# Third-party imports
from pandas.api.types import is_numeric_dtype

# Local imports
from .llm import LlamaBot

# Typing imports
from typing import Any, Dict, Union, List, Tuple

class DataHolder:
    """
    A class to manage and preprocess data for machine learning tasks. This class provides 
    functionality for setting and getting attributes, validating parameters, and performing 
    data preprocessing tasks like column normalization, type conversion, and feature-target splitting.

    Attributes:
        df (pd.DataFrame): The internal DataFrame for data processing.
        X (pd.DataFrame): The feature matrix after splitting.
        y (pd.Series): The target variable after splitting.
        csv_path (str): Path to the CSV file for loading data.
        regression_flag (bool): Whether the task is regression (1) or classification (0).
        hyperparameter_tuning (bool): Flag to enable/disable hyperparameter tuning.
        extended_models (bool): Flag to enable/disable extended models.
        cv (int): Cross-validation splits for model evaluation.
        random_state (int): Random state for reproducibility.
    """
    
    def __init__(self) -> None:
        """
        Initializes a new instance of the DataHolder class.
        """
        pass

    def get(self, name: str, value: Any = None) -> Any:
        """
        Get the value of a specified attribute from the object.

        Args:
            name (str): The name of the attribute.
            value (Any): The default value to return if the attribute does not exist.

        Returns:
            Any: The value of the attribute.
        """
        return getattr(self, name, value)

    def set(self, name: str, value: Any) -> None:
        """
        Set the value of a specified attribute in the object.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to assign to the attribute.
        """
        setattr(self, name, value)
        
        return

    def validate_parameter(self, parameter: str, correct_type: str) -> bool:
        """
        Validates the type of a specified parameter.

        Args:
            parameter (str): The name of the parameter.
            correct_type (str): The expected type of the parameter.

        Returns:
            bool: True if the type matches the expected type, otherwise False.
        """
        value = self.get(parameter)
        return type(value) == correct_type

    def validate_parameters(self, parameters: Dict[str, str]) -> List[str]:
        """
        Validates multiple parameters and checks if their types match the expected types.

        Args:
            parameters (Dict[str, str]): A dictionary where keys are parameter names 
                                         and values are the expected types.

        Returns:
            List[str]: A list of parameters that failed validation.
        """
        missing_parameters = []
        for parameter, type_ in parameters.items():
            if not self.validate_parameter(parameter, type_):
                missing_parameters.append(parameter)
        return missing_parameters

    def get_parameters(self, parameters: List[str]) -> Dict[str, Any]:
        """
        Gets the values of a list of parameters.

        Args:
            parameters (List[str]): A list of parameter names to retrieve.

        Returns:
            Dict[str, Any]: A dictionary where keys are parameter names and values are 
                            the corresponding values.
        """
        return {parameter: self.get(parameter) for parameter in parameters}

    def delete_attribute(self, name: str) -> None:
        """
        Deletes a specified attribute from the object.

        Args:
            name (str): The name of the attribute to delete.
        """
        delattr(self, name)

        return

    def delete_all_attributes(self, exceptions: List[str] = []) -> None:
        """
        Deletes all attributes of the object except those specified in the exceptions list.

        Args:
            exceptions (List[str]): A list of attribute names to exclude from deletion.
        """
        attributes = list(self.__dict__.keys())
        for attribute in attributes:
            if attribute not in exceptions:
                self.delete_attribute(attribute)

        return

    def set_modeling_parameters(self, hyperparameter_tuning: bool = False,
                                extended_models: bool = False, cv: int = 5,
                                random_state: int = 42) -> None:
        """
        Sets the modeling parameters.

        Args:
            hyperparameter_tuning (bool): Flag to enable/disable hyperparameter tuning.
            extended_models (bool): Flag to enable/disable extended models.
            cv (int): Number of cross-validation folds.
            random_state (int): Random state for reproducibility.
        """
        self.hyperparameter_tuning = hyperparameter_tuning
        self.extended_models = extended_models
        self.cv = cv
        self.random_state = random_state

        return

    def get_modeling_parameters(self) -> Dict[str, Union[bool, int, None]]:
        """
        Gets the current modeling parameters.

        Returns:
            Dict[str, Union[bool, int, None]]: A dictionary containing the modeling parameters.
        """
        return {
            'hyperparameter_tuning': self.get("hyperparameter_tuning"),
            'extended_models': self.get("extended_models"),
            'cv': self.get("cv", 5),
            'random_state': self.get("random_state", 42)
        }
        
    def validate_modeling_parameters(self) -> None:
        """
        Validates and sets the modeling parameters, filtering out None values.
        """
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

    def load_dataframe(self) -> bool:
        """
        Loads the DataFrame from a CSV file.

        Returns:
            bool: True if loading the CSV file was successful, otherwise False.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            return True
        except Exception:
            return False
    
    def normalize_column_names(self) -> bool:
        """
        Normalizes the column names of the internal DataFrame to snake_case format.
    
        This method uses an AI model (LlamaBot) to intelligently convert all column names to 
        snake_case format. If a column name is already in snake_case, it will be converted to 
        lowercase.
    
        The method interacts with a chatbot to process and return the formatted column names, 
        and then applies these names to the DataFrame's columns. If the operation is successful, 
        the column names are updated in place. If the response from the chatbot is incorrect 
        or the conversion fails, the method will return False.
    
        Returns:
            bool: True if the column names were successfully normalized, otherwise False.
        """
        
        # Instantiate a chatbot to process data names using a specified role context.
        helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)
        
        # Construct a prompt to request the chatbot to convert column names to snake_case.
        prompt = f"""
        Given the following list of strings, convert the list of strings to snake case format.
        If a column is already in snake case format, simply convert to lowercase.
        Columns: {self.df.columns}
        Do not give an explanation, only return a list of strings.
        """
        
        # Call the chatbot with the constructed prompt to get the snake_case column names.
        response = helper.call_llama_eval(prompt)
    
        # Attempt to apply the response to the DataFrame columns.
        # If the response is in an invalid format or there is an error, return False.
        try:
            self.df.columns = response
            return True  # Return True if column names were successfully updated.
        except Exception:
            return False  # Return False if there was an error applying the response.


    def convert_column_types(self) -> bool:
        """
        Infers and converts column data types in the DataFrame using LlamaBot assistance.
    
        This method leverages a language model (LlamaBot) to suggest appropriate data types 
        for each column in the internal DataFrame. It processes the DataFrame's first 10 rows 
        and infers types based on their content. Numeric columns are converted to `int` or `float`, 
        and text columns are assigned a data type of `object`. Binary columns are mapped to boolean 
        integers (0 or 1). Unsupported datetime columns are removed automatically.
    
        If the method successfully applies the inferred data types, it returns `True`. If an error occurs 
        during type conversion or the inference fails, it prints an error message and returns `False`.
    
        Returns:
            bool: `True` if column types were successfully inferred and applied, otherwise `False`.
    
        Raises:
            Exception: If the type conversion process fails, an error message is printed.
        """
        
        # Instantiate a chatbot (LlamaBot) to process data and infer data types.
        helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)
        
        # Construct a prompt to get suggested data types for each column from the language model.
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
        
        # Call LlamaBot with the constructed prompt to get a column-type mapping.
        response = helper.call_llama_eval(prompt)
        
        try:
            for column in response.keys():
                unique_values = self.df[column].unique()
    
                # Drop columns with high uniqueness (likely IDs or UUIDs that are not useful for analysis).
                if len(unique_values) >= (0.5 * len(self.df.index)):
                    self.df = self.df.drop(column, axis = 1)
    
                # Handle binary columns (with two unique values) by mapping to boolean integers.
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
    
                # Handle numeric and non-numeric columns based on the inferred type from the model.
                else:
                    numeric = is_numeric_dtype(response[column])
                    if numeric:
                        # Attempt to convert the column to a numeric type (either int or float).
                        self.df[column] = pd.to_numeric(self.df[column], errors = "coerce")
                    else:
                        # Convert the column to the inferred non-numeric data type.
                        self.df[column] = self.df[column].astype(response[column])
    
            # Inform that datetime columns cannot be handled and will be dropped.
            print("The current version cannot support date or time columns. These will be dropped automatically.")
    
            # Drop datetime columns as they are unsupported.
            self.df = self.df.select_dtypes(include = ["number", "object"])
    
            return True  # Return True if column types were successfully applied.
    
        except Exception as e:
            # Print the error message if type conversion fails.
            print("Unable to determine data types for columns. This may result in unexpected behavior for preprocessing.")
            print(e)
    
            return False  # Return False if the process fails.

    def split_features(self, y: str) -> bool:
        """
        Splits the DataFrame into features (X) and target (y).

        Args:
            y (str): The name of the target column.

        Returns:
            bool: True if the splitting was successful, otherwise False.
        """
        try:
            self.df = self.df.dropna(subset = [y])
            self.X = self.df.drop(y, axis = 1)
            self.y = self.df[y]
            return True
        except Exception:
            return False