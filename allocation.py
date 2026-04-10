from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _to_date(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _score_components(oil: Dict[str, Any], ship: Dict[str, Any]) -> Dict[str, Any]:
    quantity_mt = float(oil["quantity_mt"])
    capacity_mt = float(ship["capacity_mt"])

    if capacity_mt < quantity_mt:
        return {
            "eligible": False,
            "capacity_score": None,
            "date_score": None,
            "port_score": None,
            "contamination_penalty": None,
            "final_score": None,
        }
    
    

    capacity_score = 40.0 * (quantity_mt / capacity_mt)

    available_date = _to_date(ship["available_date"])
    delivery_deadline = _to_date(oil["delivery_deadline"])
    date_score = 30.0 if pd.notna(available_date) and pd.notna(delivery_deadline) and available_date <= delivery_deadline else 0.0

    oil_origin = oil.get("origin_port", oil.get("origin_country", ""))
    port_score = 20.0 if _normalize_text(ship.get("available_port")) == _normalize_text(oil_origin) else 10.0

    contamination_penalty = 0.0 if _normalize_text(ship.get("last_oil_type")) == _normalize_text(oil.get("oil_type")) else 100.0
    final_score = capacity_score + date_score + port_score - contamination_penalty

    return {
        "eligible": True,
        "capacity_score": capacity_score,
        "date_score": date_score,
        "port_score": port_score,
        "contamination_penalty": contamination_penalty,
        "final_score": final_score,
    }


def _format_decision_reason(decision: str, reasons: List[str]) -> str:
    unique_reasons: List[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if text and text not in unique_reasons:
            unique_reasons.append(text)

    if not unique_reasons:
        unique_reasons = ["No eligible ship found"]

    bullets = "\n".join(f"- {text}" for text in unique_reasons)
    return f"Decision: {decision}\n\nReasons:\n{bullets}"


def _build_accept_reasons(components: Dict[str, Any]) -> List[str]:
    reasons: List[str] = ["Capacity sufficient", "Port acceptable"]

    if float(components.get("date_score", 0.0)) >= 30.0:
        reasons.append("Deadline satisfied")
    else:
        reasons.append("Ship available after deadline (delay risk)")

    if float(components.get("contamination_penalty", 0.0)) == 0.0:
        reasons.append("No contamination risk")
    else:
        reasons.append("Different oil type (contamination risk) – tank cleaning required")

    return reasons


def _build_reject_reasons(components: Optional[Dict[str, Any]]) -> List[str]:
    if not components or not components.get("eligible"):
        return ["Capacity insufficient for cargo", "No eligible ship found"]

    reasons: List[str] = ["Capacity sufficient", "Port acceptable", "Final score below acceptance threshold"]

    if float(components.get("date_score", 0.0)) >= 30.0:
        reasons.append("Deadline satisfied")
    else:
        reasons.append("Ship available after deadline (delay risk)")

    if float(components.get("contamination_penalty", 0.0)) == 0.0:
        reasons.append("No contamination risk")
    else:
        reasons.append("Different oil type (contamination risk) – tank cleaning required")

    return reasons


def allocate_oil_to_ships(oils: pd.DataFrame, ships: pd.DataFrame) -> List[Dict[str, Any]]:
    if oils.empty:
        return []

    min_final_score = 20.0

    oils_working = oils.copy()
    ships_working = ships.copy()

    oils_working["quantity_mt"] = pd.to_numeric(oils_working["quantity_mt"], errors="coerce")
    ships_working["capacity_mt"] = pd.to_numeric(ships_working["capacity_mt"], errors="coerce")
    oils_working["delivery_deadline"] = pd.to_datetime(oils_working["delivery_deadline"], errors="coerce")
    ships_working["available_date"] = pd.to_datetime(ships_working["available_date"], errors="coerce")

    oils_working = oils_working.dropna(subset=["oil_id", "oil_type", "quantity_mt", "delivery_deadline", "destination_port"])
    ships_working = ships_working.dropna(subset=["ship_id", "capacity_mt", "available_date", "available_port"])

    oils_working = oils_working.sort_values(["delivery_deadline", "quantity_mt"], ascending=[True, False]).reset_index(drop=True)
    ship_pool = ships_working.to_dict(orient="records")

    allocation_rows: List[Dict[str, Any]] = []

    for _, oil_row in oils_working.iterrows():
        oil = oil_row.to_dict()
        if "origin_port" not in oil and "origin_country" in oil:
            oil["origin_port"] = oil["origin_country"]

        best_ship = None
        best_components = None
        best_score = None

        for ship in ship_pool:
            components = _score_components(oil, ship)
            if not components.get("eligible"):
                continue

            score = float(components["final_score"])
            if best_score is None or score > best_score:
                best_score = score
                best_ship = ship
                best_components = components

        should_allocate = best_ship is not None and best_score is not None and best_score >= min_final_score

        if should_allocate:
            reason = _format_decision_reason("ACCEPT", _build_accept_reasons(best_components or {}))
            allocation_rows.append(
                {
                    "oil_id": oil["oil_id"],
                    "ship_id": best_ship["ship_id"],
                    "status": "allocated",
                    "decision_reason": reason,
                    "final_score": round(float(best_score), 6),
                    "allocation_time": datetime.utcnow(),
                }
            )
            ship_pool = [ship for ship in ship_pool if str(ship.get("ship_id")) != str(best_ship.get("ship_id"))]
        else:
            reason = _format_decision_reason("REJECT", _build_reject_reasons(best_components))
            allocation_rows.append(
                {
                    "oil_id": oil["oil_id"],
                    "ship_id": None,
                    "status": "unallocated",
                    "decision_reason": reason,
                    "final_score": round(float(best_score), 6) if best_score is not None else None,
                    "allocation_time": datetime.utcnow(),
                }
            )

    return allocation_rows
