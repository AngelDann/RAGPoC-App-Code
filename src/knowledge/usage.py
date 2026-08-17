from __future__ import annotations

import time

from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.utils import timezone

from knowledge.models import ApiUsageLog
from ragpoc.config import Settings
from ragpoc.http import get as http_get


def record_usage(
    *,
    category: str,
    action: str = "",
    provider: str = "openrouter",
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    duration_ms: int | None = None,
    status: str = "success",
    error_message: str = "",
    request_count: int = 1,
    metadata: dict | None = None,
) -> None:
    """Persist one usage/log entry. Never raises — logging must not break the caller."""
    try:
        ApiUsageLog.objects.create(
            category=category,
            action=action,
            provider=provider,
            model=model or "",
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=(
                total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0)
            ),
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            status=status,
            error_message=(error_message or "")[:2000],
            request_count=request_count,
            metadata_json=metadata or {},
        )
    except Exception:
        pass


def usage_summary(days: int = 7) -> dict:
    since = timezone.now() - timezone.timedelta(days=days)
    qs = ApiUsageLog.objects.filter(created_at__gte=since)

    totals = qs.aggregate(
        requests=Count("id"),
        input_tokens=Sum("input_tokens"),
        output_tokens=Sum("output_tokens"),
        total_tokens=Sum("total_tokens"),
        cost_usd=Sum("cost_usd"),
    )
    errors = qs.filter(status="error").count()

    by_category = list(
        qs.values("category")
        .annotate(requests=Count("id"), total_tokens=Sum("total_tokens"), cost_usd=Sum("cost_usd"))
        .order_by("-requests")
    )
    by_model = list(
        qs.exclude(model="")
        .values("model")
        .annotate(requests=Count("id"), total_tokens=Sum("total_tokens"))
        .order_by("-total_tokens")[:10]
    )
    daily = list(
        qs.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(requests=Count("id"), total_tokens=Sum("total_tokens"))
        .order_by("day")
    )
    for row in daily:
        row["day"] = row["day"].isoformat()

    return {
        "since": since.isoformat(),
        "days": days,
        "requests": totals["requests"] or 0,
        "errors": errors,
        "input_tokens": totals["input_tokens"] or 0,
        "output_tokens": totals["output_tokens"] or 0,
        "total_tokens": totals["total_tokens"] or 0,
        "cost_usd": totals["cost_usd"],
        "by_category": by_category,
        "by_model": by_model,
        "daily": daily,
    }


_key_status_cache: dict = {"data": None, "ts": 0.0, "key": None}


def fetch_openrouter_key_status(settings: Settings, force: bool = False) -> dict | None:
    """Live usage/limit info for the active OpenRouter API key (cached ~30s)."""
    api_key = settings.openrouter_api_key
    if not api_key:
        return None

    now = time.time()
    cached = _key_status_cache
    if not force and cached["data"] and cached["key"] == api_key and now - cached["ts"] < 30:
        return cached["data"]

    try:
        response = http_get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get("data", {})
    except Exception as error:
        return {"error": str(error)}

    cached["data"] = data
    cached["ts"] = now
    cached["key"] = api_key
    return data
