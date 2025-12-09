import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import ARIMAModel, ARIMAXModel, SARIMAXModel, evaluate_forecast
from models.rolling_cv import RollingCV
from sklearn.preprocessing import StandardScaler
from models.regressors import (
    LinearRegressor,
    XGBoostRegressor,
    CatBoostRegressorT,
    LightGBMRegressor,
)
from anomaly.detect import detect_anomalies



def visualize_anomalies(y_true, y_pred, dates, anomalies, model_name, output_image):
    """
    Visualizes actual vs. predicted values and residuals, highlighting anomalies.
    """
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)


    ax1.plot(dates, y_true, label="Actual", color="dodgerblue", zorder=1)
    ax1.plot(dates, y_pred, label="Predicted", color="darkorange", linestyle="--", zorder=2)
    if len(anomalies) > 0:
        ax1.scatter(
            anomalies.dates, anomalies.actual_values, color="red", label="Anomalies", s=50, zorder=3
        )
    ax1.set_title(f"{model_name}: Actual vs. Predicted Trip Counts")
    ax1.set_ylabel("Trip Count")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax2.plot(dates, residuals, label="Residuals", color="mediumseagreen", zorder=1)
    if len(anomalies) > 0:
        ax2.scatter(
            anomalies.dates,
            anomalies.residuals,
            color="red",
            label="Anomalous Residuals",
            s=50,
            zorder=2,
        )
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title(f"{model_name}: Residuals")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_image)
    plt.close(fig)
    print(f"Saved anomaly visualization for {model_name} to {output_image}")


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

    initial_train_size, test_size, step = 365, 30, 30
    cv = RollingCV(initial_train_size, test_size, step)
    all_preds, all_trues, all_dates = [], [], []

    for train_indices, test_indices in cv.split(y):
        train_y, test_y = y.iloc[train_indices], y.iloc[test_indices]
        arima = ARIMAModel(order=(1, 1, 1)).fit(train_y)
        y_pred = arima.forecast(steps=len(test_y))
        all_preds.extend(y_pred)
        all_trues.extend(test_y.values)
        all_dates.extend(test_y.index)

    metrics = evaluate_forecast(np.array(all_trues), np.array(all_preds))
    anomalies = detect_anomalies(np.array(all_trues), np.array(all_preds), pd.DatetimeIndex(all_dates))
    metrics['anomaly_count'] = len(anomalies)
    metrics['anomaly_percentage'] = (len(anomalies) / len(all_trues)) * 100 if all_trues else 0

    return {
        "metrics": metrics,
        "data": {
            "trues": pd.Series(all_trues, index=all_dates),
            "preds": pd.Series(all_preds, index=all_dates),
            "dates": pd.DatetimeIndex(all_dates),
            "anomalies": anomalies,
        },
    }


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
    exog_cols = [
        "month_sin", "month_cos", "dow_sin", "dow_cos", "temp_mean", "precip",
        "precipitation_hours", "wind_speed", "wind_gusts_10m_max", "temp_range",
        "is_rainy", "is_weekend", "is_holiday", "active_closures", "new_closures",
        "ending_closures",
    ]
    _exog = df_copy[["date"] + exog_cols].copy()
    _exog["date"] = pd.to_datetime(_exog["date"], errors="coerce")
    _exog = _exog.dropna(subset=["date"]).sort_values("date").set_index("date").asfreq("D").ffill().bfill()
    for bcol in ["is_rainy", "is_weekend", "is_holiday"]:
        if bcol in _exog.columns:
            _exog[bcol] = _exog[bcol].astype(int)
    y_aligned, exog_aligned = y.align(_exog, join="inner", axis=0)
    cv_arimax = RollingCV(initial_train_size, test_size, step)
    all_preds_arimax, all_trues_arimax, all_dates_arimax = [], [], []
    cont_cols = [
        "temp_mean", "precip", "precipitation_hours", "wind_speed",
        "wind_gusts_10m_max", "temp_range", "active_closures", "new_closures",
        "ending_closures",
    ]
    for train_indices, test_indices in cv_arimax.split(y_aligned):
        train_y, test_y = y_aligned.iloc[train_indices], y_aligned.iloc[test_indices]
        exog_train, exog_test = exog_aligned.iloc[train_indices].copy(), exog_aligned.iloc[test_indices].copy()
        scaler = StandardScaler()
        exog_train[cont_cols] = scaler.fit_transform(exog_train[cont_cols])
        exog_test[cont_cols] = scaler.transform(exog_test[cont_cols])
        arimax = ARIMAXModel(order=(1, 1, 1)).fit(train_y, exog=exog_train, maxiter=1000)
        y_pred_arimax = arimax.forecast(steps=len(test_y), exog=exog_test)
        all_preds_arimax.extend(y_pred_arimax)
        all_trues_arimax.extend(test_y.values)
        all_dates_arimax.extend(test_y.index)
    metrics_arimax = evaluate_forecast(np.array(all_trues_arimax), np.array(all_preds_arimax))
    anomalies = detect_anomalies(np.array(all_trues_arimax), np.array(all_preds_arimax), pd.DatetimeIndex(all_dates_arimax))
    metrics_arimax['anomaly_count'] = len(anomalies)
    metrics_arimax['anomaly_percentage'] = (len(anomalies) / len(all_trues_arimax)) * 100 if all_trues_arimax else 0
    return {
        "metrics": metrics_arimax,
        "data": {
            "trues": pd.Series(all_trues_arimax, index=all_dates_arimax),
            "preds": pd.Series(all_preds_arimax, index=all_dates_arimax),
            "dates": pd.DatetimeIndex(all_dates_arimax),
            "anomalies": anomalies,
        },
    }


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
        "month_sin", "month_cos", "dow_sin", "dow_cos", "temp_mean", "precip",
        "precipitation_hours", "wind_speed", "wind_gusts_10m_max", "temp_range",
        "is_rainy", "is_weekend", "is_holiday", "active_closures", "new_closures",
        "ending_closures",
    ]
    _exog = df_copy[["date"] + exog_cols].copy()
    _exog["date"] = pd.to_datetime(_exog["date"], errors="coerce")
    _exog = _exog.dropna(subset=["date"]).sort_values("date").set_index("date").asfreq("D").ffill().bfill()
    for bcol in ["is_rainy", "is_weekend", "is_holiday"]:
        if bcol in _exog.columns:
            _exog[bcol] = _exog[bcol].astype(int)
    y_aligned, exog_aligned = y.align(_exog, join="inner", axis=0)
    cv_sarimax = RollingCV(initial_train_size, test_size, step)
    all_preds_sarimax, all_trues_sarimax, all_dates_sarimax = [], [], []
    cont_cols = [
        "temp_mean", "precip", "precipitation_hours", "wind_speed",
        "wind_gusts_10m_max", "temp_range", "active_closures", "new_closures",
        "ending_closures",
    ]
    for train_indices, test_indices in cv_sarimax.split(y_aligned):
        train_y, test_y = y_aligned.iloc[train_indices], y_aligned.iloc[test_indices]
        exog_train, exog_test = exog_aligned.iloc[train_indices].copy(), exog_aligned.iloc[test_indices].copy()
        scaler = StandardScaler()
        exog_train[cont_cols] = scaler.fit_transform(exog_train[cont_cols])
        exog_test[cont_cols] = scaler.transform(exog_test[cont_cols])
        sarimax = SARIMAXModel(order=(1, 1, 1), seasonal_order=seasonal_order).fit(
            train_y, exog=exog_train, maxiter=1000, disp=False
        )
        y_pred_sarimax = sarimax.forecast(steps=len(test_y), exog=exog_test)
        all_preds_sarimax.extend(y_pred_sarimax)
        all_trues_sarimax.extend(test_y.values)
        all_dates_sarimax.extend(test_y.index)
    metrics_sarimax = evaluate_forecast(np.array(all_trues_sarimax), np.array(all_preds_sarimax))
    anomalies = detect_anomalies(np.array(all_trues_sarimax), np.array(all_preds_sarimax), pd.DatetimeIndex(all_dates_sarimax))
    metrics_sarimax['anomaly_count'] = len(anomalies)
    metrics_sarimax['anomaly_percentage'] = (len(anomalies) / len(all_trues_sarimax)) * 100 if all_trues_sarimax else 0
    return {
        "metrics": metrics_sarimax,
        "data": {
            "trues": pd.Series(all_trues_sarimax, index=all_dates_sarimax),
            "preds": pd.Series(all_preds_sarimax, index=all_dates_sarimax),
            "dates": pd.DatetimeIndex(all_dates_sarimax),
            "anomalies": anomalies,
        },
    }


def eval_baseline_models(df_copy):
    """Evaluate all baseline models."""
    print("Evaluating ARIMA...")
    arima_results = eval_arima(df_copy)
    print("ARIMA Metrics:", {k: round(v, 3) for k, v in arima_results['metrics'].items()})

    print("\nEvaluating ARIMAX...")
    arimax_results = eval_arimax(df_copy)
    print("ARIMAX Metrics:", {k: round(v, 3) for k, v in arimax_results['metrics'].items()})

    print("\nEvaluating SARIMAX...")
    sarimax_results = eval_sarimax(df_copy)
    print("SARIMAX Metrics:", {k: round(v, 3) for k, v in sarimax_results['metrics'].items()})

    return {
        "ARIMA": arima_results,
        "ARIMAX": arimax_results,
        "SARIMAX": sarimax_results,
    }


def eval_regressor_models(df_copy):
    """Evaluate regressor models."""
    print("Evaluating regressor models...")
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
        "active_closures",
        "new_closures",
        "ending_closures",
    ]
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[TARGET]
    X = df[FEATURES].copy()

    initial_train_size = 365
    test_size = 30
    step = 30
    cont_cols = ["temp_mean", "precip", "wind_speed", 'active_closures', 'new_closures', 'ending_closures']
    scaler = StandardScaler()
    X[cont_cols] = scaler.fit_transform(X[cont_cols])

    initial_train_size = 365
    test_size = 30
    step = 30

    models = {
        "LinearRegressor": LinearRegressor(
            initial_train_size, test_size, step
        ),
        "XGBoost": XGBoostRegressor(
            initial_train_size,
            test_size,
            step,
            lags=[1, 7, 14],
            window_features={7: ["mean", "std"], 14: ["mean"]},
            n_estimators=100,
        ),
        "CatBoost": CatBoostRegressorT(
            initial_train_size,
            test_size,
            step,
            lags=[1, 7, 14],
            window_features={7: ["mean", "std"], 14: ["mean"]},
            iterations=100,
        ),
        "LightGBM": LightGBMRegressor(
            initial_train_size,
            test_size,
            step,
            lags=[1, 7, 14],
            window_features={7: ["mean", "std"], 14: ["mean"]},
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"  - Training {name}...")
        model.fit_predict(y, X)
        preds = model.get_predictions()
        trues = model.get_true_values()

        # Align dates for anomaly detection
        if hasattr(model, "_create_features"):
            y_feat, _ = model._create_features(y, X)
            cv_dates = []
            cv = RollingCV(initial_train_size, test_size, step)
            for _, test_indices in cv.split(y_feat):
                cv_dates.extend(y_feat.iloc[test_indices].index)
            cv_dates = pd.DatetimeIndex(cv_dates)
        else:
            cv_dates = []
            cv = RollingCV(initial_train_size, test_size, step)
            for _, test_indices in cv.split(y):
                cv_dates.extend(y.iloc[test_indices].index)
            cv_dates = pd.DatetimeIndex(cv_dates)

        metrics = evaluate_forecast(trues, preds)
        anomalies = detect_anomalies(trues, preds, cv_dates, threshold=2.5)
        metrics["anomaly_count"] = len(anomalies)
        metrics["anomaly_percentage"] = (
            len(anomalies) / len(trues) * 100 if len(trues) > 0 else 0
        )
        results[name] = {
            "metrics": metrics,
            "data": {
                "trues": trues,
                "preds": preds,
                "dates": cv_dates,
                "anomalies": anomalies,
            },
        }
    return results


def visualize_metrics(metrics, output_image="model_comparison.png"):
    baseline_metrics = metrics.get("baseline_models", {})
    regressor_metrics = metrics.get("regressor_models", {})
    all_metrics_data = []

    # Tag data correctly
    for model_name, m in baseline_metrics.items():
        m["model"] = model_name
        m["type"] = "Baseline"
        all_metrics_data.append(m)
    for model_name, m in regressor_metrics.items():
        m["model"] = model_name
        m["type"] = "Regressor"
        all_metrics_data.append(m)

    if not all_metrics_data:
        print("No metrics data to visualize.")
        return

    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_df.set_index("model", inplace=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Model Performance Comparison", fontsize=20)
    axes = axes.flatten()
    metric_names = ["RMSE", "MAE", "MAPE", "SMAPE"]

    color_map = {"Baseline": "salmon", "Regressor": "skyblue"}

    for i, metric in enumerate(metric_names):
        if metric in metrics_df.columns:
            sorted_df = metrics_df.sort_values(by=metric, ascending=False)

            colors = sorted_df["type"].map(color_map)
            sorted_df.plot(
                kind="barh",
                y=metric,
                ax=axes[i],
                color=colors,
                legend=False,
            )
            axes[i].set_title(metric)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("")
            axes[i].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image)
    plt.close(fig)
    print(f"Saved model comparison visualization to {output_image}")


def run_evaluation(data_path, output_file):
    """
    Main function to run the entire evaluation pipeline.
    """
    df = pd.read_csv(data_path)
    df_processed = process_weather_calendar(df)

    baseline_results = eval_baseline_models(df_processed)
    regressor_results = eval_regressor_models(df_processed)

    all_metrics = {
        "baseline_models": {k: v["metrics"] for k, v in baseline_results.items()},
        "regressor_models": {k: v["metrics"] for k, v in regressor_results.items()},
    }

    # Find and visualize the best baseline model
    best_baseline_model_name = min(
        all_metrics["baseline_models"],
        key=lambda k: all_metrics["baseline_models"][k]["RMSE"],
    )
    best_baseline_data = baseline_results[best_baseline_model_name]["data"]
    visualize_anomalies(
        y_true=best_baseline_data["trues"],
        y_pred=best_baseline_data["preds"],
        dates=best_baseline_data["dates"],
        anomalies=best_baseline_data["anomalies"],
        model_name=best_baseline_model_name,
        output_image="best_baseline_anomalies.png",
    )

    best_regressor_model_name = min(
        all_metrics["regressor_models"],
        key=lambda k: all_metrics["regressor_models"][k]["RMSE"],
    )
    best_regressor_data = regressor_results[best_regressor_model_name]["data"]
    visualize_anomalies(
        y_true=best_regressor_data["trues"],
        y_pred=best_regressor_data["preds"],
        dates=best_regressor_data["dates"],
        anomalies=best_regressor_data["anomalies"],
        model_name=best_regressor_model_name,
        output_image="best_regressor_anomalies.png",
    )

    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nSaved all metrics to {output_file}")

    # Generate and save the overall model comparison visualization
    visualize_metrics(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument(
        "--input_path", type=str, default='processed_data/ready/dataset_v1_pogoh_weather.csv', help="Path to the input dataset (CSV)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='evaluation_metrics.json',
        help="Path to save the evaluation metrics (JSON).",
    )
    args = parser.parse_args()
    run_evaluation(args.input_path, args.output_path)
