from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pdfplumber

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from db import get_allocated_ship_context, get_connection
from delay_prediction import predict_allocated_ship_delays


EMBED_DIM = 256


@dataclass
class ContractKnowledgeBase:
    chunks: List[str]
    embeddings: np.ndarray
    index: Optional[Any]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start += step
    return chunks


def _token_hash_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9$]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        slot = hash(token) % dim
        vector[slot] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def build_contract_kb(text: str) -> ContractKnowledgeBase:
    chunks = chunk_text(text)
    if not chunks:
        chunks = [""]

    embeddings = np.vstack([_token_hash_embedding(chunk) for chunk in chunks]).astype(np.float32)
    index = None
    if faiss is not None:
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(embeddings)
    return ContractKnowledgeBase(chunks=chunks, embeddings=embeddings, index=index)


def _retrieve_contract_context(kb: ContractKnowledgeBase, query: str, top_k: int = 5) -> str:
    query_vector = _token_hash_embedding(query).reshape(1, -1).astype(np.float32)
    k = min(top_k, len(kb.chunks))

    if k <= 0:
        return ""

    if kb.index is not None:
        _, positions = kb.index.search(query_vector, k)
        return "\n".join(kb.chunks[i] for i in positions[0] if i >= 0)

    # Fallback for environments where faiss is unavailable.
    diff = kb.embeddings - query_vector
    squared_l2 = np.sum(diff * diff, axis=1)
    positions = np.argsort(squared_l2)[:k]
    return "\n".join(kb.chunks[int(i)] for i in positions if int(i) >= 0)


def _extract_int(patterns: List[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_money(patterns: List[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            return int(float(raw))
    return None


def extract_delay_rules(kb: ContractKnowledgeBase) -> Dict[str, Dict[str, int]]:
    supplier_context = _retrieve_contract_context(kb, "supplier delay grace period penalty rate per day")
    shipping_context = _retrieve_contract_context(kb, "shipping delay grace period penalty rate per day")
    whole_context = "\n".join(kb.chunks)

    supplier_grace = _extract_int(
        [
            r"supplier.{0,80}grace\s*period.{0,20}?(\d+)\s*day",
            r"grace\s*period.{0,20}?(\d+)\s*day.{0,80}supplier",
        ],
        supplier_context + "\n" + whole_context,
    )
    shipping_grace = _extract_int(
        [
            r"shipping.{0,80}grace\s*period.{0,20}?(\d+)\s*day",
            r"grace\s*period.{0,20}?(\d+)\s*day.{0,80}shipping",
        ],
        shipping_context + "\n" + whole_context,
    )

    supplier_rate = _extract_money(
        [
            r"supplier.{0,80}(?:penalty|rate).{0,20}?\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\$\s*([0-9,]+(?:\.[0-9]+)?).{0,80}supplier.{0,80}(?:day|per\s*day)",
        ],
        supplier_context + "\n" + whole_context,
    )
    shipping_rate = _extract_money(
        [
            r"shipping.{0,80}(?:penalty|rate).{0,20}?\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\$\s*([0-9,]+(?:\.[0-9]+)?).{0,80}shipping.{0,80}(?:day|per\s*day)",
        ],
        shipping_context + "\n" + whole_context,
    )

    return {
        "supplier": {
            "grace": supplier_grace if supplier_grace is not None else 2,
            "rate": supplier_rate if supplier_rate is not None else 5000,
        },
        "shipping": {
            "grace": shipping_grace if shipping_grace is not None else 1,
            "rate": shipping_rate if shipping_rate is not None else 4000,
        },
    }


def fetch_allocated_ships_for_penalty() -> pd.DataFrame:
    try:
        with get_connection() as connection:
            query = """
                SELECT ship_id, origin_port, destination_port, capacity_mt
                FROM assignment
                WHERE status = 'allocated';
            """
            assignment_df = pd.read_sql_query(query, connection)
            if not assignment_df.empty:
                return assignment_df
    except Exception:
        pass

    context_df = get_allocated_ship_context().copy()
    if "origin_port" not in context_df.columns and "origin_country" in context_df.columns:
        context_df["origin_port"] = context_df["origin_country"]
    return context_df[["ship_id", "origin_port", "destination_port", "capacity_mt"]]


def _normalize_match_type(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if value == "exact_path":
        return "exact_path"
    if value in {"partial_match", "partial_path"}:
        return "partial_path"
    return "nearest"


def determine_delay_type(match_type: str, origin_port: str, destination_port: str) -> str:
    origin = str(origin_port or "").strip().lower()
    destination = str(destination_port or "").strip().lower()

    # System logic: route-specific and known lane delays are supplier-side, broad nearest matches are shipping-side.
    if match_type in {"exact_path", "partial_path"} and origin and destination:
        return "supplier"
    return "shipping"


def calculate_penalty(delay_days: float, grace_period: int, rate: int) -> Tuple[float, float]:
    chargeable_delay = max(0.0, float(delay_days) - float(grace_period))
    penalty = chargeable_delay * float(rate)
    return chargeable_delay, penalty


def _build_reason(
    delay_type: str,
    match_type: str,
    origin_port: str,
    destination_port: str,
    chargeable_delay: float,
    rate: int,
    penalty: float,
) -> str:
    if penalty > 0:
        delay_label = "Supplier" if delay_type == "supplier" else "Shipping"
        days_text = int(chargeable_delay) if float(chargeable_delay).is_integer() else round(chargeable_delay, 2)
        return (
            f"Penalty due to {delay_label} delay on {match_type} "
            f"({origin_port} -> {destination_port}), exceeded grace period -> "
            f"{days_text} days x ${rate}/day"
        )
    return "No penalty applied as delay is within grace period"


def run_penalty_calculation(contract_pdf_bytes: bytes, historical_csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    contract_text = extract_pdf_text(contract_pdf_bytes)
    if not contract_text:
        raise ValueError("Uploaded contract PDF did not contain readable text")

    kb = build_contract_kb(contract_text)
    rules = extract_delay_rules(kb)

    allocated_df = fetch_allocated_ships_for_penalty()
    if allocated_df.empty:
        return pd.DataFrame(columns=["ship_id", "delay_days", "penalty", "reason"]), rules

    delay_predictions = predict_allocated_ship_delays(
        allocated_df=allocated_df,
        historical_csv_path=historical_csv_path,
    )

    merged = allocated_df.merge(
        pd.DataFrame(delay_predictions)[["ship_id", "final_delay", "predicted_delay", "match_type"]],
        on="ship_id",
        how="left",
    )

    output_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        ship_id = str(row["ship_id"])
        delay_days = float(row.get("final_delay", row.get("predicted_delay", 0.0)) or 0.0)
        origin_port = str(row.get("origin_port", row.get("origin_country", "")))
        destination_port = str(row.get("destination_port", ""))
        match_type = _normalize_match_type(str(row.get("match_type", "nearest")))

        delay_type = determine_delay_type(match_type, origin_port, destination_port)
        profile = rules[delay_type]
        grace_period = int(profile["grace"])
        rate = int(profile["rate"])

        chargeable_delay, penalty = calculate_penalty(delay_days, grace_period, rate)
        reason = _build_reason(
            delay_type=delay_type,
            match_type=match_type,
            origin_port=origin_port,
            destination_port=destination_port,
            chargeable_delay=chargeable_delay,
            rate=rate,
            penalty=penalty,
        )

        output_rows.append(
            {
                "ship_id": ship_id,
                "delay_days": round(delay_days, 2),
                "penalty": round(penalty, 2),
                "reason": reason,
            }
        )

    return pd.DataFrame(output_rows), rules
