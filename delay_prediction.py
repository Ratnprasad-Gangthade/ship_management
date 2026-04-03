from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import pandas as pd

from news_risk_agent import get_news_risk_for_route

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional import safety
    Prophet = None


MIN_ROWS_FOR_PROPHET = 14


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _prepare_historical_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Historical delay CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"ds", "y", "destination_port", "capacity_mt"}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Historical delay CSV missing columns: {missing_cols}")

    if "origin_port" not in df.columns and "origin_country" not in df.columns:
        raise ValueError("Historical delay CSV missing route origin column: origin_port or origin_country")

    if "origin_port" not in df.columns and "origin_country" in df.columns:
        df["origin_port"] = df["origin_country"]

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], dayfirst=True, errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["capacity_mt"] = pd.to_numeric(df["capacity_mt"], errors="coerce")

    df = df.dropna(subset=["ds", "y", "origin_port", "destination_port", "capacity_mt"])
    return df.sort_values("ds").reset_index(drop=True)


def _predict_with_prophet_or_mean(filtered_df: pd.DataFrame, fallback_mean: float) -> float:
    if filtered_df.empty:
        return float(fallback_mean)

    filtered_df = filtered_df[["ds", "y"]].copy().dropna().sort_values("ds")
    if filtered_df.empty:
        return float(fallback_mean)

    mean_value = float(filtered_df["y"].mean())

    if Prophet is None or len(filtered_df) < MIN_ROWS_FOR_PROPHET:
        return mean_value

    try:
        model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
        model.fit(filtered_df)

        next_day = filtered_df["ds"].max() + pd.Timedelta(days=1)
        future = pd.DataFrame({"ds": [next_day]})
        forecast = model.predict(future)
        yhat = float(forecast.iloc[0]["yhat"])
        return yhat
    except Exception:
        return mean_value


def _apply_matching_rules(
    historical_df: pd.DataFrame,
    origin_port: str,
    destination_port: str,
    capacity_mt: float,
) -> Tuple[pd.DataFrame, str]:
    origin_norm = _normalize_text(origin_port)
    destination_norm = _normalize_text(destination_port)

    exact_df = historical_df[
        (historical_df["origin_port"].astype(str).str.strip().str.lower() == origin_norm)
        & (historical_df["destination_port"].astype(str).str.strip().str.lower() == destination_norm)
    ]
    if not exact_df.empty:
        return exact_df, "exact_path"

    partial_df = historical_df[
        (historical_df["destination_port"].astype(str).str.strip().str.lower() == destination_norm)
        | (historical_df["origin_port"].astype(str).str.strip().str.lower() == origin_norm)
    ]
    if not partial_df.empty:
        return partial_df, "partial_match"

    lower = float(capacity_mt) * 0.9
    upper = float(capacity_mt) * 1.1
    nearest_df = historical_df[
        (historical_df["capacity_mt"] >= lower)
        & (historical_df["capacity_mt"] <= upper)
    ]
    if not nearest_df.empty:
        return nearest_df, "capacity_band"

    return historical_df, "fallback_overall"


def predict_allocated_ship_delays(
    allocated_df: pd.DataFrame,
    historical_csv_path: str,
) -> list[Dict[str, Any]]:
    if allocated_df.empty:
        return []

    historical_df = _prepare_historical_data(historical_csv_path)
    overall_mean = float(historical_df["y"].mean())

    results: list[Dict[str, Any]] = []

    for _, row in allocated_df.iterrows():
        origin_port = row.get("origin_port", row.get("origin_country", ""))
        destination_port = row["destination_port"]
        capacity_mt = float(row["capacity_mt"])

        filtered_df, match_type = _apply_matching_rules(
            historical_df=historical_df,
            origin_port=origin_port,
            destination_port=destination_port,
            capacity_mt=capacity_mt,
        )

        predicted_delay = _predict_with_prophet_or_mean(filtered_df, overall_mean)
        news_risk = get_news_risk_for_route(str(origin_port), str(destination_port))
        news_risk_score = float(news_risk.get("news_risk_score", 0.0) or 0.0)
        final_delay = round(float(predicted_delay) + news_risk_score, 3)

        results.append(
            {
                "ship_id": str(row["ship_id"]),
                "path": f"{origin_port} -> {destination_port}",
                "predicted_delay": round(float(predicted_delay), 3),
                "news_risk_score": round(news_risk_score, 3),
                "final_delay": final_delay,
                "risk_signals": news_risk.get("risk_signals", {}),
                "news_summary": news_risk.get("news_summary", ""),
                "match_type": match_type,
            }
        )

    return results
