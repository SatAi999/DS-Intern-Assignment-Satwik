import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class DataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None

    def handle_missing_values(self, df):
        """Handle missing values using mean imputation for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df

    def remove_low_variance_features(self, df, threshold=0.0):
        """Remove features with low variance (optional for now)."""
        selector = VarianceThreshold(threshold=threshold)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = pd.DataFrame(selector.fit_transform(df[numeric_cols]),
                                  columns=numeric_cols[selector.get_support()])
        df = df[df_numeric.columns]  # Filter the original dataframe
        return df

    def scale_features(self, df):
        """Standardize numerical features using StandardScaler."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
    
    # Handling 'unknown' values
    def preprocess_data(df):
    # Replace 'unknown' with NaN for any object type column
      for col in df.select_dtypes(include=['object']).columns:
         df[col] = df[col].replace('unknown', np.nan)

    # Optionally, impute or drop missing values (for example, using median for numeric columns)
      df.fillna(df.median(), inplace=True)  # For numerical columns
      df.fillna(df.mode().iloc[0], inplace=True)  # For categorical columns (like 'unknown' replaced)

      return df



    def preprocess(self, df):
        """Main method to perform preprocessing steps."""
        print("Handling missing values...")
        df = self.handle_missing_values(df)

        print("Scaling numeric features...")
        df = self.scale_features(df)

        # remove low-variance features
        df = self.remove_low_variance_features(df, threshold=0.01)

        print("Preprocessing completed.")
        return df
