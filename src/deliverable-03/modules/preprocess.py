from .chatbot import *
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_bool_dtype
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split

class DataFramePreprocessor:
    def __init__(self, csv_path: str):
        try:        
            self.df = pd.read_csv(csv_path)
            self.helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)
            print("Normalizing column names")
            self.normalize_column_names()
            print("Converting column types")
            self.convert_column_types()
        except Exception as e:
            print(f"Unable to load data at {csv_path}")
            print(e)
            self.df = pd.DataFrame()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df
        
    def normalize_column_names(self) -> None:
        prompt = f"""
        Given the following list of strings, convert the list of strings to snake case format.
        If a column is already in snake case format, simply convert to lowercase.
        Columns: {self.df.columns}
        Do not give an explanation, only return a list of strings.
        """
        
        response = self.helper.call_llama_eval(prompt)
        
        try:
            self.df.columns = response
        except:
            print("Unable to normalize columns, using original column names instead.")

        return

    def convert_column_types(self) -> None:       
        prompt = f"""
        Determine which data types should be used for the following Pandas dataframe:
        
        DataFrame: {self.df.head(50)}
        
        Do not give an explanation.
        If a column has more than two classes, do not return assign a value of int.
        If a column has empty strings but is otherwise numeric, assign an appropriate numeric data type.
        Convert datetime columns as necessary.
        Only use int, float, datetime64[ns], or object as possible values.
        Provide only a dictionary with keys as column names and values as the assigned pandas datatypes.
        """
        
        response = self.helper.call_llama_eval(prompt)
        # print(response)
        try:
            for column in response.keys():
                unique_values = len(self.df[column].value_counts())
                # If number of unique values is equal to number of rows, the column is useless
                # Most commonly observed with columns containing ID attributes
                if unique_values == len(self.df.index):
                    self.df = self.df.drop(column, axis = 1)
                elif unique_values == 2:
                    prompt = f"""
                    Give a mapping to convert the values to a boolean variable.
                    values: {self.df[column].value_counts().index.tolist()}
                    
                    Use the original value as the key, do not convert the original value's data type.
                    Use the appropriate boolean value cast as an integer for the value.
                    Return a dictionary.
                    """
                    mapping_dict = self.helper.call_llama_eval(prompt)

                    self.df[column] = self.df[column].map(mapping_dict).astype(np.int64)
                else:
                    numeric = is_numeric_dtype(response[column])
                    if numeric:
                        self.df[column] = pd.to_numeric(self.df[column], errors = "coerce")
                    else:
                        self.df[column] = self.df[column].astype(response[column])
            print("The current version cannot support date or time columns. These will be dropped automatically.")
            self.df = self.df.select_dtypes(include = ['number', 'object'])
        except Exception as e:
            print("Unable to determine data types for columns. This may result in unexpected behavior for preprocessing.")
            print(e)

        return


    def set_machine_learning_task(self, task_type: bool) -> None:
        self.task_type = task_type # 0 for classification, 1 for regression

        return

    def split_features(self, y: str) -> None:
        self.df = self.df.dropna(subset = [y])
        self.X = self.df.drop(y, axis = 1)
        self.y = self.df[y]

        return

    def validate_target_variable(self) -> int:
        prompt = f"""
        Determine if the following column contains continuous or categorical data.
        
        Column: {self.y}
        
        Return 0 for categorical or 1 for continuous.
        """
        
        response = self.helper.call_llama_eval(prompt)

        # 0 means match, 1 means classification on continuous data, -1 means regression on categorical data
        return response - self.task_type

    def bin_continuous_data(self) -> pd.Series:

        
        pass
    
    def train_test_split(self, test_size: float = 0.2) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size)
        return

    def impute_central_tendency(self, column: str, dtype: bool) -> None:
        if dtype:
            mean = self.X_train[column].mean()
            self.X_train[column] = self.X_train[column].fillna(mean)
            self.X_test[column] = self.X_test[column].fillna(mean)
        else:
            mode = self.X_train[column].mode().iloc[0]
            self.X_train[column] = self.X_train[column].fillna(mode)
            self.X_test[column] = self.X_test[column].fillna(mode)

        return

    def prune_outliers(self, column: str, dtype: bool) -> None:
        if dtype:
            iqr = self.X_train[column].quantile(0.75) - self.X_train[column].quantile(0.25)
            median = self.X_train[column].median()
            self.X_train = self.X_train[np.abs((self.X_train[column] - median) / iqr) <= 1.5]
            self.X_test = self.X_test[np.abs((self.X_test[column] - median) / iqr) <= 1.5]
        else:
            value_counts = self.X_train[column].value_counts(normalize = True)
            non_outlier_values = value_counts[value_counts > 0.003].index.tolist()
            self.X_train = self.X_train[self.X_train[column].isin(non_outlier_values)]
            self.X_test = self.X_test[self.X_test[column].isin(non_outlier_values)]

        return

    def detect_and_relabel_outliers(self, column: str, dtype: bool) -> None:
        if dtype:
            # Check if there are placeholder negative values, typically there is only 1, if any
            negative_values = self.X[self.X[column] < 0].value_counts()
            if len(negative_values) == 1:
                negative_value = negative_values.index[0]
                self.X_train[column] = self.X_train[column].replace({negative_value: np.nan})
                self.X_test[column] = self.X_test[column].replace({negative_value: np.nan})

            # Check there is a large positive placeholder value, if any
            max_value = self.X[column].max()
            # If the max value is an order of magnitude greater than the 99th percentile
            # of the data without the max value it is probably a placeholder value
            if max_value > (self.X[self.X[column] < max_value][column].quantile(0.99) * 10):
                self.X_train[column] = self.X_train[column].replace({max_value: np.nan})
                self.X_test[column] = self.X_test[column].replace({max_value: np.nan})
        else:
            # Merge potential similar strings
            value_counts = self.X[column].value_counts()
            values = value_counts.index.tolist()
            map_dict = {}

            for i, value_one in enumerate(values):
                for value_two in values[i + 1:]:
                    if SequenceMatcher(value_one, value_two).ratio() >= 0.9:
                        if value_two not in map_dict.keys():
                            map_dict[value_two] = value_one

            if map_dict:
                self.X_train[column] = self.X_train[column].replace(map_dict)
                self.X_test[column] = self.X_test[column].replace(map_dict)

            # Identify potential null strings and replace with single null string
            null_strings = ["Unknown", "Not Applicable", "N/A", "None", "Missing", "NULL", " "]
            null_dict = {"": np.nan}

            for value in values:
                similarities = np.array([SequenceMatcher(value.lower(),
                                                         null_string.lower()).ratio()
                                         for null_string in null_strings])
                if similarities.max() == 1:
                    null_dict[value] = np.nan

            if null_dict:
                self.X_train[column] = self.X_train[column].replace(null_dict)
                self.X_train[column] = self.X_train[column].fillna("N/A")
                self.X_test[column] = self.X_test[column].replace(null_dict)
                self.X_test[column] = self.X_test[column].fillna("N/A")

        return
        
    def clean_data(self, strictness: int = 1) -> None:
        # Calculate the missingness for columns to determine course of action
        missingness = (self.df.isnull().sum() / len(self.df.index)).to_dict()

        for column in self.X.columns:
            # If the variable is binary, no cleaning is necessary
            if len(self.X[column].value_counts().index) == 2:
                continue

            # Determine column type
            numeric = is_numeric_dtype(self.X[column].dtype)

            # # If not a numeric column, remove the column if the maximum value count for the column is 1
            # # This is because the feature is not useful and is likely an ID type attribute
            # if not numeric:
            #     if self.X[column].value_counts().max():
            #         self.X_train = self.X_train.drop(column, axis = 1)
            #         self.X_test = self.X_test.drop(column, axis = 1)
            
            # Maximum strictness, use LLM to try to clean up values
            if strictness > 1:
                # If missingness is greater than 50%, eliminate the column for lack of data
                if missingness[column] > 0.5:
                    self.X_train = self.X_train.drop(column, axis = 1)
                    self.X_test = self.X_test.drop(column, axis = 1)
                    
                    continue
                    
                else:
                    self.detect_and_relabel_outliers(column, numeric)

            # Medium strictness, prune outliers naively
            if strictness > 0:
                self.prune_outliers(column, numeric)

            # Always impute null values with measures of central tendency
            self.impute_central_tendency(column, numeric)
            # print(self.X_train[column])

        # Ensure y data is the same shape as X data
        self.y_train = self.y_train[self.y_train.index.isin(self.X_train.index)]
        self.y_test = self.y_test[self.y_test.index.isin(self.X_test.index)]
        
        return