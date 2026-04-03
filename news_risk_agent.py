from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests


NEWSAPI_URL = "https://newsapi.org/v2/everything"
RISK_KEYWORDS = {
    "strikes": ["strike", "labor protest", "dockworker strike"],
    "wars": ["war", "conflict", "military escalation", "sanction"],
    "port_congestion": ["port congestion", "berth delay", "vessel queue"],
    "weather_disruptions": ["storm", "cyclone", "hurricane", "flood", "typhoon"],
}


def _score_text(text: str) -> tuple[float, Dict[str, int]]:
    normalized = text.lower()
    signal_hits: Dict[str, int] = {signal: 0 for signal in RISK_KEYWORDS}
    score = 0.0

    for signal, keywords in RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                signal_hits[signal] += 1

        if signal_hits[signal] > 0:
            if signal == "wars":
                score += 2.0
            elif signal == "weather_disruptions":
                score += 1.5
            else:
                score += 1.0

    return min(score, 7.0), signal_hits


def _top_two_headline_summary(articles: List[Dict[str, Any]]) -> str:
    headlines: List[str] = []
    for article in articles:
        title = str(article.get("title", "")).strip()
        if title:
            headlines.append(title)
        if len(headlines) == 2:
            break

    if not headlines:
        return "No recent headlines found for this route."

    if len(headlines) == 1:
        return f"Top news: {headlines[0]}"

    return f"Top news: 1) {headlines[0]} 2) {headlines[1]}"


def get_news_risk_for_route(origin_port: str, destination_port: str) -> Dict[str, Any]:
    api_key = os.getenv("NEWSAPI_KEY", "your_api_key")
    if not api_key or api_key == "your_api_key":
        return {
            "news_risk_score": 0.0,
            "risk_signals": {signal: 0 for signal in RISK_KEYWORDS},
            "news_summary": "NewsAPI key not configured. Using placeholder 'your_api_key'.",
            "articles_scanned": 0,
        }

    seven_days_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    query = f'("{origin_port}" OR "{destination_port}") AND (oil OR shipping OR tanker OR port)'

    params = {
        "q": query,
        "from": seven_days_ago,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 30,
        "apiKey": api_key,
    }

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as error:
        return {
            "news_risk_score": 0.0,
            "risk_signals": {signal: 0 for signal in RISK_KEYWORDS},
            "news_summary": f"NewsAPI request failed: {error}",
            "articles_scanned": 0,
        }

    articles: List[Dict[str, Any]] = payload.get("articles", []) or []
    aggregate_score = 0.0
    aggregate_hits: Dict[str, int] = {signal: 0 for signal in RISK_KEYWORDS}

    for article in articles:
        title = str(article.get("title", ""))
        description = str(article.get("description", ""))
        content = f"{title} {description}"

        article_score, article_hits = _score_text(content)
        aggregate_score += article_score
        for signal, count in article_hits.items():
            aggregate_hits[signal] += count

    normalized_score = min(round(aggregate_score / max(len(articles), 1), 3), 7.0)
    headline_summary = _top_two_headline_summary(articles)

    return {
        "news_risk_score": normalized_score,
        "risk_signals": aggregate_hits,
        "news_summary": f"Scanned {len(articles)} articles. {headline_summary}",
        "articles_scanned": len(articles),
    }
