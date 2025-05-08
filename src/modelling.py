# modeling.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class ModelTrainer:
    def __init__(self, model=None):
        """
        Initializes the model trainer with a default model.
        If no model is passed, a RandomForestRegressor is used by default.
        """
        self.model = model if model else RandomForestRegressor()

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Trains the model on the provided data and splits into train/test datasets.

        Args:
            X (pd.DataFrame): Features for training.
            y (pd.Series): Target variable.
            test_size (float): Proportion of the data to be used for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: The trained model and the test dataset (X_test, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def predict(self, X_test):
        """
        Makes predictions using the trained model.

        Args:
            X_test (pd.DataFrame): Features for which predictions are made.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """
        Evaluates the model performance using RMSE, MAE, and R².

        Args:
            y_true (pd.Series or np.ndarray): Actual target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: A dictionary containing the RMSE, MAE, and R².
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

