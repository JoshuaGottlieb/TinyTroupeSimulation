# Absolute imports
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# Local imports
from .dataholder import *
from .llm import *

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Typing imports
from typing import List

class DataFramePreprocessor:
    """
    A class designed to handle preprocessing of csv files to convert them into
    Pandas DataFrames using NumPy, Pandas, and a Llama 3 8b LLM model.

    Preparation is designed for classical regression or classification tasks using the scikit-learn library.
    """
    
    def __init__(self, dataholder: DataHolder):
        """
        Initializes the data preprocessing class by loading a CSV file and preparing the DataFrame.

        If the CSV cannot be loaded, an empty DataFrame is used instead.
    
        Args:
            csv_path (str): The file path to the CSV file.
    
        Raises:
            Prints an error message if the file cannot be loaded or if any step fails.
        """
        self.dataholder = dataholder
        self.helper = LlamaBot(role_context = """
            You are an AI assistant designed to help preprocess data.
            You do not provide explanations and only return answers in the requested format.
            """, temperature = 0.1)

    def validate_target_variable(self) -> int:
        """
        Validates whether the target variable matches the expected task type.
    
        Returns:
            int: 0 if the target matches the expected task type,
                 1 if it is classification on continuous data,
                -1 if it is regression on categorical data.
        """
        # If there are only 2 unique values, the target is categorical (0)
        if len(self.dataholder.y.unique()) == 2:
            response = 0
        else:
            # Prompt the language model to determine if the target is categorical (0) or continuous (1)
            prompt = f"""
            Determine if the following column contains continuous or categorical data.
            
            Column: {self.dataholder.y}
            
            Return 0 for categorical or 1 for continuous.
            """
            
            response = self.helper.call_llama_eval(prompt)
    
        # Compare with expected task type (e.g., 0 = classification, 1 = regression)
        # Return difference: 0 = match, 1 or -1 = mismatch
        return response - self.dataholder.regression_flag
        
    def bin_continuous_data(self, num_bins: int = 3) -> List[str]:
        """
        Discretizes a continuous target variable into categorical bins using quantile-based binning.
    
        This method:
            - Uses pandas `qcut` to divide the target variable into quantile-based bins.
            - Handles duplicate bin edges gracefully.
            - Stores numeric bin indices in `dataholder.y`.
            - Returns a list of human-readable interval labels corresponding to the bins.
    
        Args:
            num_bins (int): The number of quantile bins to create. Default is 3.
    
        Returns:
            List[str]: A list of string labels representing the bin intervals, or an empty list if binning fails.
        """
        try:
            # Apply quantile-based binning to the target variable
            binned_data, bin_edges = pd.qcut(
                self.dataholder.y,
                q = num_bins,
                retbins = True,
                labels = False,
                duplicates = 'drop'  # Avoids errors from non-unique bin edges
            )
    
            # Generate readable interval labels for each bin
            interval_labels = [
                f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                for i in range(len(bin_edges) - 1)
            ]
    
            # Replace numeric bin indices with human-readable labels
            labeled_data = binned_data.map(lambda i: interval_labels[i] if pd.notna(i) else np.nan)
    
            # Update target column to use bin indices (still numerical, float for consistency)
            self.dataholder.y = pd.Series(
                labeled_data,
                index = self.dataholder.y.index,
                dtype = "float",
                name = self.dataholder.y.name
            )
            return interval_labels
    
        except Exception as e:
            # Log the error if binning fails
            print(f"Error during binning: {e}")
            return []
   
    def train_test_split(self, test_size: float = 0.2) -> None:
        """
        Splits the dataset into training and testing sets.
    
        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split.
                                         Default is 0.2 (20%).
        """
        # Use scikit-learn's train_test_split to split features and target
        self.dataholder.X_train, self.dataholder.X_test, \
            self.dataholder.y_train, self.dataholder.y_test = train_test_split(
            self.dataholder.X, self.dataholder.y, test_size = test_size
        )
    
        return

    def impute_central_tendency(self, column: str, dtype: bool) -> None:
        """
        Imputes missing values in a column using central tendency.
    
        Args:
            column (str): The name of the column to impute.
            dtype (bool): If True, imputes with mean (numerical column).
                          If False, imputes with mode (categorical column).
        """
        if dtype:
            # Numerical column: fill missing values with the mean from the training set
            mean = self.dataholder.X_train[column].mean()
            self.dataholder.X_train[column] = self.dataholder.X_train[column].fillna(mean)
            self.dataholder.X_test[column] = self.dataholder.X_test[column].fillna(mean)
        else:
            # Categorical column: fill missing values with the mode from the training set
            mode = self.dataholder.X_train[column].mode().iloc[0]
            self.dataholder.X_train[column] = self.dataholder.X_train[column].fillna(mode)
            self.dataholder.X_test[column] = self.dataholder.X_test[column].fillna(mode)
    
        return

    def prune_outliers(self, column: str, dtype: bool) -> None:
        """
        Prunes outliers from a column in the training and testing datasets.
    
        Args:
            column (str): The name of the column to prune outliers from.
            dtype (bool): If True, applies outlier pruning for numerical data (using IQR method).
                          If False, applies outlier pruning for categorical data (based on frequency threshold).
        """
        if dtype:
            # Numerical column: Use the Interquartile Range (IQR) method to identify and remove outliers
            iqr = self.dataholder.X_train[column].quantile(0.75) - self.dataholder.X_train[column].quantile(0.25)
            median = self.dataholder.X_train[column].median()
    
            # If IQR is 0, all values are essentially the same — drop the column
            if iqr == 0:
                self.dataholder.X_train = self.dataholder.X_train.drop(column, axis = 1)
                self.dataholder.X_test = self.dataholder.X_test.drop(column, axis = 1)
            else:
                # Calculate how many rows would be considered outliers
                is_outlier = np.abs((self.dataholder.X_train[column] - median) / iqr) > 1.5
                X_train_length = len(self.dataholder.X_train.index)
                outlier_length = is_outlier.sum()
    
                # If removing outliers would drop more than 1% of data, keep the column as-is
                # These values are probably real data due to a non-normal distribution as opposed to outliers
                if (X_train_length - outlier_length) / X_train_length < 0.99:
                    return
                else:
                    # Keep only rows within 1.5 * IQR of the median
                    self.dataholder.X_train = self.dataholder.X_train[
                    np.abs((self.dataholder.X_train[column] - median) / iqr) <= 1.5
                    ]
                    self.dataholder.X_test = self.dataholder.X_test[
                    np.abs((self.dataholder.X_test[column] - median) / iqr) <= 1.5
                    ]
        else:
            # Categorical column: Remove infrequent categories
            value_counts = self.dataholder.X_train[column].value_counts(normalize = True)
    
            # Only keep values that appear in more than 0.3% of the training data
            non_outlier_values = value_counts[value_counts > 0.003].index.tolist()
            self.dataholder.X_train = self.dataholder.X_train[self.dataholder.X_train[column].isin(non_outlier_values)]
            self.dataholder.X_test = self.dataholder.X_test[self.dataholder.X_test[column].isin(non_outlier_values)]
    
        return

    def detect_and_relabel_outliers(self, column: str, dtype: bool) -> None:
        """
        Detects and relabels outliers or placeholder values in a specified column.
    
        Args:
            column (str): The name of the column to detect and relabel outliers.
            dtype (bool): If True, handles numerical columns (looking for negative and large positive outliers).
                          If False, handles categorical columns (looking for similar string values or null-like strings).
        """
        if dtype:
            # For numerical columns, check for negative placeholder values (usually a single negative outlier)
            negative_values = self.dataholder.X[self.dataholder.X[column] < 0].value_counts()
            if len(negative_values) == 1:
                negative_value = negative_values.index[0]
                # Replace the negative placeholder value with NaN
                self.dataholder.X_train[column] = self.dataholder.X_train[column].replace({negative_value: np.nan})
                self.dataholder.X_test[column] = self.dataholder.X_test[column].replace({negative_value: np.nan})
    
            # For numerical columns, check for a large positive placeholder value (greater than 10 times next largest value)
            max_value = self.dataholder.X[column].max()
            if max_value > (self.dataholder.X[self.dataholder.X[column] < max_value][column].max() * 10):
                # Replace the large max value with NaN as it likely represents a placeholder value
                self.dataholder.X_train[column] = self.dataholder.X_train[column].replace({max_value: np.nan})
                self.dataholder.X_test[column] = self.dataholder.X_test[column].replace({max_value: np.nan})
        else:
            # For categorical columns, find similar string values and merge them
            value_counts = self.dataholder.X[column].value_counts()
            values = value_counts.index.tolist()

            # Apply string merging if the number of unique values is less than 50 due to time complexity
            if len(values) <= 50:
                map_dict = {}
        
                # Use SequenceMatcher to merge strings with high similarity (>= 90%)
                for i, value_one in enumerate(values):
                    for value_two in values[i + 1:]:
                        if SequenceMatcher(value_one, value_two).ratio() >= 0.9:
                            if value_two not in map_dict.keys():
                                map_dict[value_two] = value_one
        
                # Replace similar strings based on the map_dict
                if map_dict:
                    self.dataholder.X_train[column] = self.dataholder.X_train[column].replace(map_dict)
                    self.dataholder.X_test[column] = self.dataholder.X_test[column].replace(map_dict)
    
            # For categorical columns, identify and replace potential null-like strings with NaN
            null_strings = ["Unknown", "Not Applicable", "N/A", "None", "Missing", "NULL", " "]
            null_dict = {"": np.nan}
    
            # Compare each value in the column to null-like strings using SequenceMatcher
            for value in values:
                similarities = np.array([SequenceMatcher(value.lower(), null_string.lower()).ratio()
                                         for null_string in null_strings])
                if similarities.max() == 1:
                    null_dict[value] = np.nan
    
            # Replace identified null-like values with NaN and fill NaNs with a standard value ("N/A")
            if null_dict:
                self.dataholder.X_train[column] = self.dataholder.X_train[column].replace(null_dict)
                self.dataholder.X_train[column] = self.dataholder.X_train[column].fillna("N/A")
                self.dataholder.X_test[column] = self.dataholder.X_test[column].replace(null_dict)
                self.dataholder.X_test[column] = self.dataholder.X_test[column].fillna("N/A")
    
        return
        
    def clean_data(self, strictness: int = 1) -> None:
        """
        Cleans the dataset by handling missing values, outliers, and imputation based on strictness level.
    
        Args:
            strictness (int, optional): Controls the level of cleaning applied:
                     - 0: Basic cleaning (only imputation).
                     - 1: Medium strictness (prune outliers and impute).
                     - 2: High strictness (detect and relabel outliers in addition to pruning and imputation).
                                        Default is 1 (Medium strictness).
        """
        # Calculate missingness for each column (percentage of missing values)
        missingness = (self.dataholder.df.isnull().sum() / len(self.dataholder.df.index)).to_dict()
    
        # Iterate over each column in the feature set (X)
        for column in self.dataholder.X.columns:
            # Skip binary columns (no cleaning needed for binary variables)
            if len(self.dataholder.X[column].value_counts().index) == 2:
                continue
    
            # Determine if the column is numerical or categorical
            numeric = is_numeric_dtype(self.dataholder.X[column].dtype)
            
            # High strictness (strictness > 1): Use techniques to clean data and remove columns with too much missing data
            if strictness > 1:
                # If the missingness is greater than 50%, drop the column as it lacks sufficient data
                if missingness[column] > 0.5:
                    self.dataholder.X_train = self.dataholder.X_train.drop(column, axis = 1)
                    self.dataholder.X_test = self.dataholder.X_test.drop(column, axis = 1)
                    continue # Skip further processing for this column
                else:
                    # Detect and relabel outliers if missingness is not too high
                    self.detect_and_relabel_outliers(column, numeric)
                
            # Medium strictness (strictness > 0): Prune outliers naively
            if strictness > 0:
                self.prune_outliers(column, numeric)

                # Numeric column may be dropped if IQR = 0
                if column not in self.dataholder.X_train.columns:
                    continue # Skip further processing for this column
    
            # Always impute missing values using the central tendency (mean for numerical, mode for categorical)
            self.impute_central_tendency(column, numeric)
        
        # Ensure the target (y) data aligns with the cleaned feature (X) data by matching indices
        self.dataholder.y_train = self.dataholder.y_train[self.dataholder.y_train.index.isin(self.dataholder.X_train.index)]
        self.dataholder.y_test = self.dataholder.y_test[self.dataholder.y_test.index.isin(self.dataholder.X_test.index)]
    
        return

    def standardize_and_encode(self) -> None:
        """
        Creates preprocessor that standardizes numerical features and applies one-hot encoding to categorical features.
    
        Categorical and binary features are one-hot encoded, with the first category dropped to prevent multicollinearity.
        For numerical columns, it standardizes them using StandardScaler.
        
        If the task is classification, the target variable is label-encoded.
        """
        # Get the unique values in each column of the training dataset
        unique_values = [(column, len(self.dataholder.X_train[column].unique()))
                         for column in self.dataholder.X_train.columns]
    
        # Identify binary columns (those with only two unique values)
        # Identify categorical columns to apply one-hot encoding, combining with binary columns
        binary_columns = [column for column, num_values in unique_values if num_values == 2]
        one_hot_columns = list(set(self.dataholder.X_train.select_dtypes("object").columns.tolist() + binary_columns))
    
        # Identify numerical columns that are not part of the one-hot encoded columns
        numerical_columns = [column for column in self.dataholder.X_train.select_dtypes("number")
                             if column not in one_hot_columns]
    
        # Combine both transformations using ColumnTransformer
        self.dataholder.preprocessor = ColumnTransformer(
            transformers = [
                ("ohe", OneHotEncoder(drop = "first"), one_hot_columns), # Apply one-hot encoding to categorical columns
                ("ssc", StandardScaler(), numerical_columns) # Apply standardization to numerical columns
            ], remainder = "passthrough"
        )
    
        # If classification, encode target variable using LabelEncoder
        if not self.dataholder.regression_flag:
            self.dataholder.target_encoder = LabelEncoder()
            self.dataholder.y_train = pd.Series(self.dataholder.target_encoder.fit_transform(self.dataholder.y_train),
                                                name = self.dataholder.y.name)
            self.dataholder.y_test = pd.Series(self.dataholder.target_encoder.transform(self.dataholder.y_test),
                                               name = self.dataholder.y.name)
    
        return


    def preprocess(self) -> bool:
        try:
            self.train_test_split()
            self.clean_data(strictness = self.dataholder.cleaning_strictness)
            self.standardize_and_encode()
            return True
        except Exception as e:
            print (e)
            return False

    
