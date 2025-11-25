import argparse

import sys

import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from models import ARIMAModel, ARIMAXModel, SARIMAXModel, evaluate_forecast
from models.rolling_cv import RollingCV
from sklearn.preprocessing import StandardScaler
from models.regressors import LinearRegressor, XGBoostRegressor, CatBoostRegressorT
import json


def process_weather_calendar(df):
    """Process the input DataFrame for modeling."""
    df_copy = df.copy()
    df_copy["trip_count"] = df_copy["trip_count"].ffill().bfill()
    lag_cols = ["trip_count_lag_1", "trip_count_lag_7", "trip_count_lag_14"]
    for col in lag_cols:
        df_copy[col] = df_copy[col].fillna(df_copy["trip_count"])
    # parse dates safely
    if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
        df_dates = pd.to_datetime(df_copy["date"], errors="coerce")
    else:
        df_dates = df_copy["date"]

    ordered = df_copy.assign(__date=df_dates).sort_values("__date")

    # shift by 1 day so current day's rolling stats use only prior days
    s = ordered["trip_count"]
    s_shift = s.shift(1)
    # compute rolling statistics
    roll_mean_7 = s_shift.rolling(window=7, min_periods=1).mean()
    roll_std_7 = s_shift.rolling(window=7, min_periods=1).std(ddof=0)
    roll_mean_14 = s_shift.rolling(window=14, min_periods=1).mean()
    roll_std_14 = s_shift.rolling(window=14, min_periods=1).std(ddof=0)

    # re-align to original row order
    roll_mean_7 = roll_mean_7.sort_index()
    roll_std_7 = roll_std_7.sort_index()
    roll_mean_14 = roll_mean_14.sort_index()
    roll_std_14 = roll_std_14.sort_index()

    # overwrite rolling feature columns to reflect updated trip_count
    df_copy["trip_count_rolling_mean_7"] = roll_mean_7
    df_copy["trip_count_rolling_std_7"] = roll_std_7
    df_copy["trip_count_rolling_mean_14"] = roll_mean_14
    df_copy["trip_count_rolling_std_14"] = roll_std_14

    # the very first row has no prior day; set remaining NaN (if any) to 0 to avoid leakage
    check_cols = [
        "trip_count_rolling_mean_7",
        "trip_count_rolling_std_7",
        "trip_count_rolling_mean_14",
        "trip_count_rolling_std_14",
    ]
    df_copy[check_cols] = df_copy[check_cols].fillna(0)

    return df_copy


def eval_arima(df_copy):
    _df = df_copy
    y = _df[["date", "trip_count"]].copy()
    y["date"] = pd.to_datetime(y["date"], errors="coerce")
    y = (
        y.dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")["trip_count"]
        .astype(float)
    )
    y = y.asfreq("D").ffill().bfill()

    # rolling CV parms
    initial_train_size = 365
    test_size = 30
    step = 30

    cv = RollingCV(initial_train_size, test_size, step)
    all_preds = []
    all_trues = []

    for train_indices, test_indices in cv.split(y):
        train_y = y.iloc[train_indices]
        test_y = y.iloc[test_indices]

        arima = ARIMAModel(order=(1, 1, 1)).fit(train_y)
        y_pred = arima.forecast(steps=len(test_y))

        all_preds.extend(y_pred)
        all_trues.extend(test_y.values)

    metrics = evaluate_forecast(np.array(all_trues), np.array(all_preds))

    return metrics


def eval_arimax(df_copy):
    initial_train_size = 365
    test_size = 30
    step = 30
    _df = df_copy
    y = _df[["date", "trip_count"]].copy()
    y["date"] = pd.to_datetime(y["date"], errors="coerce")
    y = (
        y.dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")["trip_count"]
        .astype(float)
    )
    y = y.asfreq("D").ffill().bfill()
    # exogenous variables
    exog_cols = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "temp_mean",
        "precip",
        "precipitation_hours",
        "wind_speed",
        "wind_gusts_10m_max",
        "temp_range",
        "is_rainy",
        "is_weekend",
        "is_holiday",
    ]
    _exog = df_copy[["date"] + exog_cols].copy()
    _exog["date"] = pd.to_datetime(_exog["date"], errors="coerce")
    _exog = _exog.dropna(subset=["date"]).sort_values("date").set_index("date")
    _exog = _exog.asfreq("D").ffill().bfill()

    for bcol in ["is_rainy", "is_weekend", "is_holiday"]:
        if bcol in _exog.columns:
            _exog[bcol] = _exog[bcol].astype(int)

    y_aligned, exog_aligned = y.align(_exog, join="inner", axis=0)

    cv_arimax = RollingCV(initial_train_size, test_size, step)
    all_preds_arimax = []
    all_trues_arimax = []

    # scaling
    cont_cols = [
        "temp_mean",
        "precip",
        "precipitation_hours",
        "wind_speed",
        "wind_gusts_10m_max",
        "temp_range",
    ]

    for train_indices, test_indices in cv_arimax.split(y_aligned):
        train_y = y_aligned.iloc[train_indices]
        test_y = y_aligned.iloc[test_indices]

        exog_train = exog_aligned.iloc[train_indices].copy()
        exog_test = exog_aligned.iloc[test_indices].copy()

        scaler = StandardScaler()
        exog_train[cont_cols] = scaler.fit_transform(exog_train[cont_cols])
        exog_test[cont_cols] = scaler.transform(exog_test[cont_cols])

        arimax = ARIMAXModel(order=(1, 1, 1)).fit(
            train_y, exog=exog_train, maxiter=1000
        )
        y_pred_arimax = arimax.forecast(steps=len(test_y), exog=exog_test)

        all_preds_arimax.extend(y_pred_arimax)
        all_trues_arimax.extend(test_y.values)

    metrics_arimax = evaluate_forecast(
        np.array(all_trues_arimax), np.array(all_preds_arimax)
    )

    return metrics_arimax


def eval_sarimax(df_copy):
    initial_train_size = 365
    test_size = 30
    step = 30
    seasonal_order = (1, 0, 0, 7)
    _df = df_copy
    y = _df[["date", "trip_count"]].copy()
    y["date"] = pd.to_datetime(y["date"], errors="coerce")
    y = (
        y.dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")["trip_count"]
        .astype(float)
    )
    y = y.asfreq("D").ffill().bfill()
    exog_cols = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "temp_mean",
        "precip",
        "precipitation_hours",
        "wind_speed",
        "wind_gusts_10m_max",
        "temp_range",
        "is_rainy",
        "is_weekend",
        "is_holiday",
    ]
    _exog = df_copy[["date"] + exog_cols].copy()
    _exog["date"] = pd.to_datetime(_exog["date"], errors="coerce")
    _exog = _exog.dropna(subset=["date"]).sort_values("date").set_index("date")
    _exog = _exog.asfreq("D").ffill().bfill()

    for bcol in ["is_rainy", "is_weekend", "is_holiday"]:
        if bcol in _exog.columns:
            _exog[bcol] = _exog[bcol].astype(int)

    y_aligned, exog_aligned = y.align(_exog, join="inner", axis=0)

    cv_sarimax = RollingCV(initial_train_size, test_size, step)
    all_preds_sarimax = []
    all_trues_sarimax = []
    # scaling
    cont_cols = [
        "temp_mean",
        "precip",
        "precipitation_hours",
        "wind_speed",
        "wind_gusts_10m_max",
        "temp_range",
    ]

    for train_indices, test_indices in cv_sarimax.split(y_aligned):
        train_y = y_aligned.iloc[train_indices]
        test_y = y_aligned.iloc[test_indices]
        exog_train = exog_aligned.iloc[train_indices].copy()
        exog_test = exog_aligned.iloc[test_indices].copy()

        scaler = StandardScaler()
        exog_train[cont_cols] = scaler.fit_transform(exog_train[cont_cols])
        exog_test[cont_cols] = scaler.transform(exog_test[cont_cols])

        sarimax = SARIMAXModel(order=(1, 1, 1), seasonal_order=seasonal_order).fit(
            train_y, exog=exog_train, maxiter=1000, disp=False
        )
        y_pred_sarimax = sarimax.forecast(steps=len(test_y), exog=exog_test)

        all_preds_sarimax.extend(y_pred_sarimax)
        all_trues_sarimax.extend(test_y.values)

    metrics_sarimax = evaluate_forecast(
        np.array(all_trues_sarimax), np.array(all_preds_sarimax)
    )
    return metrics_sarimax


def eval_baseline_models(df_copy):
    print("Evaluating ARIMA...")
    arima_metrics = eval_arima(df_copy)
    print("Evaluating ARIMAX...")
    arimax_metrics = eval_arimax(df_copy)
    print("Evaluating SARIMAX...")
    sarimax_metrics = eval_sarimax(df_copy)

    return {
        "ARIMA": arima_metrics,
        "ARIMAX": arimax_metrics,
        "SARIMAX": sarimax_metrics,
    }


def eval_linear_regressor(df_copy):
    df = df_copy.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    TARGET = "trip_count"
    FEATURES = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "temp_mean",
        "precip",
        "wind_speed",
        "is_holiday",
        "is_weekend",
    ]
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[TARGET]
    X = df[FEATURES].copy()

    initial_train_size = 365
    test_size = 30
    step = 30
    cont_cols = ["temp_mean", "precip", "wind_speed"]
    scaler = StandardScaler()
    X[cont_cols] = scaler.fit_transform(X[cont_cols])

    # linear regressor
    linear_reg = LinearRegressor(initial_train_size, test_size, step)
    linear_reg.fit_predict(y, X)
    linear_preds = linear_reg.get_predictions()
    true_values = linear_reg.get_true_values()
    linear_metrics = evaluate_forecast(true_values, linear_preds)
    return linear_metrics


def eval_xgboost_regressor(df_copy):
    df = df_copy.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    TARGET = "trip_count"
    FEATURES = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "temp_mean",
        "precip",
        "wind_speed",
        "is_holiday",
        "is_weekend",
    ]
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[TARGET]
    X = df[FEATURES].copy()
    initial_train_size = 365
    test_size = 30
    step = 30

    xgb_reg = XGBoostRegressor(
        initial_train_size,
        test_size,
        step,
        lags=[1, 7, 14],
        window_features={7: ["mean", "std"], 14: ["mean"]},
        n_estimators=100,
    )
    xgb_reg.fit_predict(y, X)
    xgb_preds = xgb_reg.get_predictions()
    xgb_true_values = xgb_reg.get_true_values()
    xgb_metrics = evaluate_forecast(xgb_true_values, xgb_preds)
    return xgb_metrics


def eval_catboost_regressor(df_copy):
    df = df_copy.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    TARGET = "trip_count"
    FEATURES = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "temp_mean",
        "precip",
        "wind_speed",
        "is_holiday",
        "is_weekend",
    ]
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[TARGET]
    X = df[FEATURES].copy()
    initial_train_size = 365
    test_size = 30
    step = 30

    catboost_reg = CatBoostRegressorT(
        initial_train_size,
        test_size,
        step,
        lags=[1, 7, 14],
        window_features={7: ["mean", "std"], 14: ["mean"]},
        iterations=100,
    )
    catboost_reg.fit_predict(y, X)
    catboost_preds = catboost_reg.get_predictions()
    catboost_true_values = catboost_reg.get_true_values()
    catboost_metrics = evaluate_forecast(catboost_true_values, catboost_preds)
    return catboost_metrics


def eval_regressor_models(df_copy):
    print("Evaluating LinearRegressor...")
    linear_metrics = eval_linear_regressor(df_copy)
    print("Evaluating XGBoostRegressor...")
    xgb_metrics = eval_xgboost_regressor(df_copy)
    print("Evaluating CatBoostRegressor...")
    catboost_metrics = eval_catboost_regressor(df_copy)

    return {
        "LinearRegressor": linear_metrics,
        "XGBoostRegressor": xgb_metrics,
        "CatBoostRegressor": catboost_metrics,
    }


def run_evaluation(data_path, output_file):
    df = pd.read_csv(data_path)
    df_processed = process_weather_calendar(df)
    print("Data processed for evaluation.")

    baseline_metrics = eval_baseline_models(df_processed)
    regressor_metrics = eval_regressor_models(df_processed)

    metrics = {}

    for model_name, model_metrics in {**baseline_metrics, **regressor_metrics}.items():
        # Convert numpy types to native Python types for JSON serialization
        metrics[model_name] = {
            "RMSE": float(model_metrics.get("RMSE", 0.0)),
            "MAE": float(model_metrics.get("MAE", 0.0)),
            "MAPE": float(model_metrics.get("MAPE", 0.0)),
            "SMAPE": float(model_metrics.get("SMAPE", 0.0)),
        }

    # Write metrics to JSON file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_file}")

    print("Evaluation Metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: {model_metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation of forecasting models."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file containing date, trip_count, and features.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics.json",
        help="Path to save the metrics JSON file.",
    )
    args = parser.parse_args()

    run_evaluation(args.data_path, args.output_file)
