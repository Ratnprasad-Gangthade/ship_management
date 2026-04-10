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


def _normalize_contract_text(contract_text: str) -> str:
    lines: List[str] = []
    for raw_line in str(contract_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _split_numbered_sections(contract_text: str) -> Dict[str, str]:
    normalized = _normalize_contract_text(contract_text)
    section_pattern = re.compile(
        r"(?im)^\s*(\d+)\.?\s+([A-Z][A-Z\s\-–]+?)\s*$",
        flags=re.MULTILINE,
    )

    matches = list(section_pattern.finditer(normalized))
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = re.sub(r"\s+", " ", str(match.group(2)).strip().upper())
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        body = str(normalized[start:end]).strip()
        sections[title] = body

    return sections


def _extract_clause_block(contract_text: str, heading: str) -> str:
    sections = _split_numbered_sections(contract_text)
    target = re.sub(r"\s+", " ", str(heading).strip().upper())

    if target in sections:
        return sections[target]

    for title, body in sections.items():
        if target in title or title in target:
            return body

    normalized = _normalize_contract_text(contract_text)
    escaped_heading = re.escape(str(heading).strip())
    pattern = rf"(?is)(?:^|\n)\s*(?:\d+\.)?\s*{escaped_heading}\s*(.*?)(?=\n\s*(?:\d+\.)\s+[A-Z][A-Z\s\-–]+\s*$|\Z)"
    match = re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if match:
        return str(match.group(1)).strip()

    return ""


def _extract_clause_numeric_rules(clause_text: str, party: str) -> Tuple[int | None, int | None]:
    grace = _extract_int(
        [
            r"grace\s*period\s*[:\-]?\s*(\d+)\s*day",
            rf"{party}.{{0,100}}grace\s*period.{{0,30}}(\d+)\s*day",
        ],
        clause_text,
    )

    rate = _extract_money(
        [
            r"penalty\s*rate\s*[:\-]?\s*\$\s*([0-9,]+(?:\.[0-9]+)?)\s*per\s*day",
            r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*per\s*day",
            rf"{party}.{{0,120}}\$\s*([0-9,]+(?:\.[0-9]+)?)",
        ],
        clause_text,
    )

    return grace, rate


def _build_clause_keywords(clause_text: str, defaults: List[str]) -> List[str]:
    normalized = str(clause_text or "").lower()
    keywords: List[str] = []
    for item in defaults:
        token = str(item).strip().lower()
        if token and token not in keywords:
            keywords.append(token)

    candidates = [
        "cargo",
        "loading",
        "readiness",
        "documentation",
        "supplier",
        "vessel",
        "shipping",
        "arrival",
        "port",
        "congestion",
        "weather",
        "storm",
        "strike",
        "war",
        "breakdown",
        "route",
    ]
    for candidate in candidates:
        if candidate in normalized and candidate not in keywords:
            keywords.append(candidate)

    return keywords


def extract_delay_rules(kb: ContractKnowledgeBase, contract_text: str) -> Dict[str, Dict[str, Any]]:
    supplier_context = _retrieve_contract_context(kb, "supplier delay grace period penalty rate per day")
    shipping_context = _retrieve_contract_context(kb, "shipping delay grace period penalty rate per day")
    whole_context = _normalize_contract_text(contract_text)

    supplier_clause = _extract_clause_block(whole_context, "SUPPLIER DELAY CLAUSE")
    shipping_clause = _extract_clause_block(whole_context, "SHIPPING DELAY CLAUSE")
    responsibility_clause = _extract_clause_block(whole_context, "DELAY RESPONSIBILITY")

    supplier_keywords = _build_clause_keywords(
        supplier_clause,
        ["supplier", "cargo", "loading", "readiness", "documentation", "oil grade"],
    )
    shipping_keywords = _build_clause_keywords(
        shipping_clause,
        ["shipping", "vessel", "arrival", "port", "congestion", "weather", "strike", "war", "breakdown"],
    )

    supplier_clause_grace, supplier_clause_rate = _extract_clause_numeric_rules(supplier_clause, "supplier")
    shipping_clause_grace, shipping_clause_rate = _extract_clause_numeric_rules(shipping_clause, "shipping")

    supplier_grace = supplier_clause_grace if supplier_clause_grace is not None else _extract_int(
        [
            r"supplier.{0,80}grace\s*period.{0,20}?(\d+)\s*day",
            r"grace\s*period.{0,20}?(\d+)\s*day.{0,80}supplier",
        ],
        supplier_context + "\n" + whole_context,
    )
    shipping_grace = shipping_clause_grace if shipping_clause_grace is not None else _extract_int(
        [
            r"shipping.{0,80}grace\s*period.{0,20}?(\d+)\s*day",
            r"grace\s*period.{0,20}?(\d+)\s*day.{0,80}shipping",
        ],
        shipping_context + "\n" + whole_context,
    )

    supplier_rate = supplier_clause_rate if supplier_clause_rate is not None else _extract_money(
        [
            r"supplier.{0,80}(?:penalty|rate).{0,20}?\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\$\s*([0-9,]+(?:\.[0-9]+)?).{0,80}supplier.{0,80}(?:day|per\s*day)",
        ],
        supplier_context + "\n" + whole_context,
    )
    shipping_rate = shipping_clause_rate if shipping_clause_rate is not None else _extract_money(
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
            "keywords": supplier_keywords,
            "clause_text": supplier_clause,
        },
        "shipping": {
            "grace": shipping_grace if shipping_grace is not None else 1,
            "rate": shipping_rate if shipping_rate is not None else 4000,
            "keywords": shipping_keywords,
            "clause_text": shipping_clause,
        },
        "_responsibility": {
            "method": "contract_clause_keywords_with_cause_signals",
            "fallback_party": "supplier",
            "clause_text": responsibility_clause,
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


def _keyword_score(text: str, keywords: List[str]) -> float:
    normalized = str(text or "").lower()
    score = 0.0
    for keyword in keywords:
        token = str(keyword).strip().lower()
        if not token:
            continue
        if token in normalized:
            score += 1.0
    return score


def _build_cause_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in ["delay_cause", "delay_reason", "news_summary", "match_type"]:
        value = row.get(field)
        if value is not None:
            text = str(value).strip()
            if text:
                parts.append(text)

    risk_signals = row.get("risk_signals", {})
    if isinstance(risk_signals, dict):
        for signal, count in risk_signals.items():
            try:
                numeric = int(count)
            except Exception:
                numeric = 0
            if numeric > 0:
                parts.append(str(signal))

    return " ".join(parts).strip()


def determine_delay_type(row: Dict[str, Any], rules: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    fallback_party = str(rules.get("_responsibility", {}).get("fallback_party", "supplier")).strip().lower()
    if fallback_party not in {"supplier", "shipping"}:
        fallback_party = "supplier"

    explicit_party = str(row.get("responsible_party", "")).strip().lower()
    if explicit_party in {"supplier", "shipping"}:
        return explicit_party, "responsible_party field"

    match_type = _normalize_match_type(str(row.get("match_type", "nearest")))
    if match_type == "nearest":
        return fallback_party, "contract fallback party (nearest match)"

    cause_text = _build_cause_text(row)
    supplier_keywords = [str(k) for k in rules.get("supplier", {}).get("keywords", [])]
    shipping_keywords = [str(k) for k in rules.get("shipping", {}).get("keywords", [])]

    supplier_score = _keyword_score(cause_text, supplier_keywords)
    shipping_score = _keyword_score(cause_text, shipping_keywords)

    if shipping_score > supplier_score:
        return "shipping", "contract shipping clause keywords"
    if supplier_score > shipping_score:
        return "supplier", "contract supplier clause keywords"

    return fallback_party, "contract fallback party"


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
    attribution_basis: str,
) -> str:
    if penalty > 0:
        delay_label = "Supplier" if delay_type == "supplier" else "Shipping"
        days_text = int(chargeable_delay) if float(chargeable_delay).is_integer() else round(chargeable_delay, 2)
        return (
            f"Penalty due to {delay_label} delay on {match_type} "
            f"({origin_port} -> {destination_port}), exceeded grace period -> "
            f"{days_text} days x ${rate}/day (responsibility via {attribution_basis})"
        )
    return "No penalty applied as delay is within grace period"


def run_penalty_calculation(contract_pdf_bytes: bytes, historical_csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    contract_text = extract_pdf_text(contract_pdf_bytes)
    if not contract_text:
        raise ValueError("Uploaded contract PDF did not contain readable text")

    kb = build_contract_kb(contract_text)
    rules = extract_delay_rules(kb, contract_text)

    allocated_df = fetch_allocated_ships_for_penalty()
    if allocated_df.empty:
        return pd.DataFrame(columns=["ship_id", "delay_days", "penalty", "reason"]), rules

    delay_predictions = predict_allocated_ship_delays(
        allocated_df=allocated_df,
        historical_csv_path=historical_csv_path,
    )

    merged = allocated_df.merge(
        pd.DataFrame(delay_predictions)[["ship_id", "final_delay", "predicted_delay", "match_type", "news_summary", "risk_signals"]],
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

        delay_type, attribution_basis = determine_delay_type(row.to_dict(), rules)
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
            attribution_basis=attribution_basis,
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
