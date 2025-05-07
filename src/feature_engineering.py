import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np

class FeatureEngineer:
    """
    A class for performing feature engineering on a Pandas DataFrame.
    """
    def __init__(self):
        """
        Initializes the FeatureEngineer class.
        """
        self.scaler = StandardScaler()
        self.poly = None  # Initialize PolynomialFeatures instance

    def create_interaction_terms(self, df, features_list, degree=2, interaction_only=True, include_bias=False):
        """
        Creates interaction terms between specified features in a DataFrame using PolynomialFeatures.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features_list (list): A list of feature names to create interactions from.
            degree (int, optional): The degree of the polynomial features. Default is 2.
            interaction_only (bool, optional): Whether to include only interaction terms. Default is True.
            include_bias (bool, optional): Whether to include a bias term. Default is False.

        Returns:
            pd.DataFrame: The DataFrame with added interaction terms.  Returns original df if
                          len(features_list) < 2.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(features_list, list):
            raise TypeError("features_list must be a list")
        if len(features_list) < 2:
            print("Warning: features_list must contain at least two features to create interaction terms. Returning original DataFrame.")
            return df

        try:
            # Initialize PolynomialFeatures with parameters
            self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

            # Create interaction features
            interaction_df = pd.DataFrame(self.poly.fit_transform(df[features_list]),
                                            columns=self.poly.get_feature_names_out(features_list))

            # Drop original columns to avoid duplication
            interaction_df.drop(columns=features_list, inplace=True, errors='ignore')
            # Concatenate the new features with the original DataFrame
            df = pd.concat([df.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)
            return df
        except Exception as e:
            print(f"Error in create_interaction_terms: {e}")
            return df # Return original df in case of error

    def engineer_features(self, df):
        """
        Creates domain-specific features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the added domain-specific features.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        try:
            # Handle potential division by zero by adding a small constant
            if 'zone1_humidity' in df and 'zone1_temperature' in df:
                df['humidity_temp_ratio_zone1'] = df['zone1_humidity'] / (df['zone1_temperature'] + 1e-5)

            if 'dew_point' in df and 'atmospheric_pressure' in df:
                df['dew_pressure_interaction'] = df['dew_point'] * df['atmospheric_pressure']

            if 'zone1_temperature' in df and 'zone5_temperature' in df:
                df['temp_diff_zone1_zone5'] = df['zone1_temperature'] - df['zone5_temperature']

            if 'equipment_energy_consumption' in df:
                df['is_high_energy'] = (df['equipment_energy_consumption'] > df['equipment_energy_consumption'].median()).astype(int)
            return df
        except Exception as e:
            print(f"Error in engineer_features: {e}")
            return df

    def scale_numeric(self, df, numeric_cols):
        """
        Scales numeric features in a DataFrame using StandardScaler.

        Args:
            df (pd.DataFrame): The input DataFrame.
            numeric_cols (list): A list of column names to scale.

        Returns:
            pd.DataFrame: The DataFrame with scaled numeric features.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(numeric_cols, list):
            raise TypeError("numeric_cols must be a list")

        # 1. Remove non-numeric columns from the list to prevent errors.
        valid_numeric_cols = []
        for col in numeric_cols:
            if col in df and pd.api.types.is_numeric_dtype(df[col]):
                valid_numeric_cols.append(col)
            elif col in df:
                print(f"Warning: Column '{col}' is not numeric and will be skipped for scaling.")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame and will be skipped for scaling.")

        if not valid_numeric_cols:
            print("Warning: No numeric columns found for scaling. Returning original DataFrame.")
            return df

        try:
            # 2. Scale the numeric columns
            df[valid_numeric_cols] = self.scaler.fit_transform(df[valid_numeric_cols])
            return df
        except Exception as e:
            print(f"Error in scale_numeric: {e}")
            return df
