"""Daily spend anomaly detection over usage.db."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta

from .store import UsageStore

MIN_HISTORY_DAYS = 3


@dataclass(frozen=True)
class SpendAnomaly:
    model: str
    day: str
    day_total: float
    trailing_median: float | None
    ratio: float | None
    rule: str


def detect_spend_anomalies(
    store: UsageStore,
    day: date,
    window_days: int = 14,
    multiple: float = 3.0,
    ceiling: float | None = None,
) -> list[SpendAnomaly]:
    """Compare each model's `day` spend to its trailing `window_days` median.

    Flags a model when its day total is at least `multiple` times the
    trailing median, or exceeds the absolute `ceiling` (if given). A model
    with fewer than MIN_HISTORY_DAYS days with a reported cost in the trailing
    window has no median, so only the ceiling applies. A median of 0 has no
    ratio either, so `multiple` does not apply and only the ceiling can flag
    the day. Days with no reported cost are skipped.
    """
    since = (day - timedelta(days=window_days)).isoformat()
    until = (day + timedelta(days=1)).isoformat()
    target = day.isoformat()

    day_totals: dict[str, float] = {}
    history: dict[str, list[float]] = {}
    for row in store.daily_cost_by_model(since=since, until=until):
        model = row["model"]
        cost = row["cost"]
        if cost is None:
            continue
        if row["day"] == target:
            day_totals[model] = cost
        else:
            history.setdefault(model, []).append(cost)

    anomalies = []
    for model, day_total in day_totals.items():
        trailing = history.get(model, [])
        median: float | None = None
        ratio: float | None = None
        if len(trailing) >= MIN_HISTORY_DAYS:
            median = statistics.median(trailing)
            if median > 0:
                ratio = day_total / median

        rules = []
        if ratio is not None and ratio >= multiple:
            rules.append("multiple")
        if ceiling is not None and day_total > ceiling:
            rules.append("ceiling")
        if not rules:
            continue

        anomalies.append(
            SpendAnomaly(
                model=model,
                day=target,
                day_total=day_total,
                trailing_median=median,
                ratio=ratio,
                rule="+".join(rules),
            )
        )

    return anomalies
