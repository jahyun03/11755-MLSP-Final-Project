import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from typing import Optional
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from models.rolling_cv import RollingCV

class BaseRegressor:
    """
    Base class for regressor models with rolling cross-validation.
    
    Parameters
    ----------
    initial_train_size : int
        Size of the initial training set for rolling CV.
    test_size : int
        Size of the test set (forecast horizon) for rolling CV.
    step : int
        Step size for rolling CV.
    """
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1):
        self.cv = RollingCV(initial_train_size, test_size, step)
        self.model = None
        self.predictions = []
        self.true_values = []

    def fit_predict(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        """
        Fit the model and generate predictions using rolling cross-validation.
        
        Parameters
        ----------
        y : pd.Series
            The target time series.
        exog : Optional[pd.DataFrame]
            Exogenous variables.
        
        Returns
        -------
        self
        """
        if exog is None:
            raise ValueError("Exogenous variables are required for regressor models.")
            
        X = exog.values
        y_vals = y.values
        
        for train_indices, test_indices in self.cv.split(X):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y_vals[train_indices], y_vals[test_indices]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            
            self.predictions.extend(preds)
            self.true_values.extend(y_test)
            
        return self

    def get_predictions(self) -> np.ndarray:
        """Returns the out-of-sample predictions from the rolling CV."""
        return np.array(self.predictions)

    def get_true_values(self) -> np.ndarray:
        """Returns the true values corresponding to the predictions."""
        return np.array(self.true_values)

    def evaluate(self) -> float:
        """
        Evaluates the model using Root Mean Squared Error (RMSE).
        
        Returns
        -------
        float
            The RMSE of the predictions.
        """
        if not self.predictions:
            raise ValueError("You must call fit_predict before evaluating the model.")
        return np.sqrt(mean_squared_error(self.true_values, self.predictions))

class LinearRegressor(BaseRegressor):
    """Linear Regression model with rolling CV."""
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1, **kwargs):
        super().__init__(initial_train_size, test_size, step)
        self.model = LinearRegression(**kwargs)

class RidgeRegressor(BaseRegressor):
    """Ridge Regression model with rolling CV."""
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1, alpha: float = 1.0, **kwargs):
        super().__init__(initial_train_size, test_size, step)
        self.model = Ridge(alpha=alpha, **kwargs)

class LassoRegressor(BaseRegressor):
    """Lasso Regression model with rolling CV."""
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1, alpha: float = 1.0, **kwargs):
        super().__init__(initial_train_size, test_size, step)
        self.model = Lasso(alpha=alpha, **kwargs)

class XGBoostRegressor(BaseRegressor):
    """XGBoost Regressor model with rolling CV."""
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1, lags: list = None, window_features: dict = None, **kwargs):
        super().__init__(initial_train_size, test_size, step)
        self.model = XGBRegressor(**kwargs)
        self.lags = lags if lags is not None else []
        self.window_features = window_features if window_features is not None else {}

    def _create_features(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        X = exog.copy() if exog is not None else pd.DataFrame(index=y.index)
        
        # Lag features
        for lag in self.lags:
            X[f'lag_{lag}'] = y.shift(lag)
            
        # Window features
        for window, funcs in self.window_features.items():
            for func in funcs:
                X[f'window_{func}_{window}'] = y.shift(1).rolling(window=window).agg(func)

        X.dropna(inplace=True)
        y = y[X.index]
        
        return y, X

    def fit_predict(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        y, X = self._create_features(y, exog)
        
        y_vals = y.values
        X_vals = X.values
        
        for train_indices, test_indices in self.cv.split(X):
            X_train, X_test = X_vals[train_indices], X_vals[test_indices]
            y_train, y_test = y_vals[train_indices], y_vals[test_indices]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            
            self.predictions.extend(preds)
            self.true_values.extend(y_test)
            
        return self

class CatBoostRegressorT(BaseRegressor):
    """CatBoost Regressor model with rolling CV."""
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1, lags: list = None, window_features: dict = None, **kwargs):
        super().__init__(initial_train_size, test_size, step)
        self.model = CatBoostRegressor(verbose=0, **kwargs)
        self.lags = lags if lags is not None else []
        self.window_features = window_features if window_features is not None else {}

    def _create_features(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        X = exog.copy() if exog is not None else pd.DataFrame(index=y.index)
        
        # Lag features
        for lag in self.lags:
            X[f'lag_{lag}'] = y.shift(lag)
            
        # Window features
        for window, funcs in self.window_features.items():
            for func in funcs:
                X[f'window_{func}_{window}'] = y.shift(1).rolling(window=window).agg(func)

        X.dropna(inplace=True)
        y = y[X.index]
        
        return y, X

    def fit_predict(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        y, X = self._create_features(y, exog)
        
        y_vals = y.values
        X_vals = X.values
        
        for train_indices, test_indices in self.cv.split(X):
            X_train, X_test = X_vals[train_indices], X_vals[test_indices]
            y_train, y_test = y_vals[train_indices], y_vals[test_indices]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            
            self.predictions.extend(preds)
            self.true_values.extend(y_test)
            
        return self

__all__ = ['LinearRegressor', 'RidgeRegressor', 'LassoRegressor', 'XGBoostRegressor', 'CatBoostRegressorT']
