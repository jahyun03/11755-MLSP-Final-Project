import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ARIMAModel:
    """ARIMA without exogenous variables"""
    
    def __init__(
        self, 
        order: tuple[int, int, int] = (1, 1, 1),
    ):
        """
        Initialize model.
        
        Parameters
        ----------
        order : tuple[int, int, int] 
            (p, d, q) order of the ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, y: pd.Series, **kwargs):
        """
        Fit ARIMA model to time series.
        
        Parameters
        ----------
        y : pd.Series
            Time series to fit
        **kwargs
            Additional arguments passed to SARIMAX.fit()
            
        Returns
        -------
        self
        """
        self.model = ARIMA(
            y,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # fitting parameters
        fit_kwargs = {
            'method': 'lbfgs',
            'maxiter': 200,
        }
        fit_kwargs.update(kwargs)
        
        self.fitted_model = self.model.fit(method_kwargs=fit_kwargs)
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecast values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted prior to prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def forecast(self, steps: int = 1) -> pd.Series:
        """
        Generate forecasts as Series.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        pd.Series
            Forecast values
        """
        return pd.Series(self.predict(steps))
    
    # Akaike Information Criterion of the fitted model
    def get_aic(self) -> float:
        return self.fitted_model.aic if self.fitted_model else None
    
    # Bayesian Information Criterion of fitted model
    def get_bic(self) -> float:
        """Get BIC of fitted model."""
        return self.fitted_model.bic if self.fitted_model else None

    
class ARIMAXModel:
    """ARIMAX with exogenous variables using ARIMA"""
    
    def __init__(self, order: tuple[int, int, int] = (1, 1, 1),):
        """
        Initialize ARIMAX model.
        
        Parameters
        ----------
        order : tuple[int, int, int]
            (p, d, q) order of the ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.exog_columns = None
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None, **kwargs):
        """
        Fit ARIMAX model to time series.
        
        Parameters
        ----------
        y : pd.Series
            Time series to fit
        exog : Optional[pd.DataFrame]
            Exogenous variables
        **kwargs
            Additional arguments passed to ARIMA.fit()
            
        Returns
        -------
        self
        """
        if exog is not None:
            self.exog_columns = exog.columns.tolist()
        
        self.model = ARIMA(
            y,
            exog=exog,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # fitting parameters
        fit_kwargs = {
            'method': 'lbfgs',
            'maxiter': 200,
        }
        fit_kwargs.update(kwargs)

        self.fitted_model = self.model.fit(method_kwargs=fit_kwargs)
        return self
    
    def predict(self, steps: int = 1, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
        exog : Optional[pd.DataFrame]
            Future exogenous variables (required if model was fit with exog)
            
        Returns
        -------
        np.ndarray
            Forecast values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted prior to prediction")
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return forecast.values
    
    def forecast(self, steps: int = 1, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts as Series.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
        exog : Optional[pd.DataFrame]
            Future exogenous variables
            
        Returns
        -------
        pd.Series
            Forecast values
        """
        return pd.Series(self.predict(steps, exog))
    
    # akaike information criterion
    def get_aic(self) -> float:
        return self.fitted_model.aic if self.fitted_model else None
    
    # bayesian information criterion
    def get_bic(self) -> float:
        return self.fitted_model.bic if self.fitted_model else None

class SARIMAXModel:
    """SARIMAX model that extends ARIMAX to include seasonal components."""
    
    def __init__(
        self, 
        order: tuple[int, int, int] = (1, 1, 1), 
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        """
        Initialize SARIMAX model.
        
        Parameters
        ----------
        order : tuple[int, int, int]
            (p, d, q) order of the ARIMA model.
        seasonal_order : tuple[int, int, int, int]
            (P, D, Q, s) seasonal order of the ARIMA model.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.exog_columns = None
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None, **kwargs):
        """
        Fit SARIMAX model to time series.
        
        Parameters
        ----------
        y : pd.Series
            Time series to fit.
        exog : Optional[pd.DataFrame]
            Exogenous variables.
        **kwargs
            Additional arguments passed to ARIMA.fit().
            
        Returns
        -------
        self
        """
        if exog is not None:
            self.exog_columns = exog.columns.tolist()
        
        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        
        # Fitting parameters
        fit_kwargs = {
            'method': 'lbfgs',
            'maxiter': 200,
        }
        fit_kwargs.update(kwargs)

        try:
            self.fitted_model = self.model.fit(**fit_kwargs)
        except TypeError:
            # Fallback for older statsmodels versions
            self.fitted_model = self.model.fit(method_kwargs=fit_kwargs)
            
        return self
    
    def predict(self, steps: int = 1, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : Optional[pd.DataFrame]
            Future exogenous variables (required if model was fit with exog).
            
        Returns
        -------
        np.ndarray
            Forecast values.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted prior to prediction")
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return forecast.values
    
    def forecast(self, steps: int = 1, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts as Series.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : Optional[pd.DataFrame]
            Future exogenous variables.
            
        Returns
        -------
        pd.Series
            Forecast values.
        """
        return pd.Series(self.predict(steps, exog))
    
    def get_aic(self) -> float:
        """Get AIC of fitted model."""
        return self.fitted_model.aic if self.fitted_model else None
    
    def get_bic(self) -> float:
        """Get BIC of fitted model."""
        return self.fitted_model.bic if self.fitted_model else None

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate forecast evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict[str, float]
        Dictionary with RMSE, MAE, MAPE, SMAPE
    """
    y_pred = np.maximum(y_pred, 0)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # mean absolute percentage error
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # symmetric mean absolute percentage error)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() > 0:
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'SMAPE': smape
    }

__all__ = [
    'ARIMAModel', 
    'ARIMAXModel', 
    'SARIMAXModel',
    'evaluate_forecast',
]
