"""基于真实 Token Usage 的可配置成本估算。

本模块不内置“最新价格”，因为价格会变化且不同账户/缓存层可能不同。管理员可在 V3
runtime 的 pricing.json 中配置经自己核实的价格；找不到模型时明确返回 unknown。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CostEstimate:
    status: str
    model_name: str | None
    currency: str | None = None
    estimated_cost: float | None = None
    pricing_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PricingCatalog:
    """读取用户维护的模型单价，单位是每一百万 Token。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def estimate(
        self,
        model_name: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        *,
        usage_available: bool,
        mocked_usage: bool = False,
    ) -> CostEstimate:
        if mocked_usage:
            return CostEstimate("mocked_usage", model_name)
        if not usage_available or input_tokens is None or output_tokens is None:
            return CostEstimate("usage_unavailable", model_name)
        config = self.read()
        model = (config.get("models") or {}).get(model_name or "")
        if not isinstance(model, dict):
            return CostEstimate("unknown_pricing", model_name)
        try:
            value = (
                input_tokens * float(model["input_per_million"])
                + output_tokens * float(model["output_per_million"])
            ) / 1_000_000
        except (KeyError, TypeError, ValueError):
            return CostEstimate("invalid_pricing", model_name)
        return CostEstimate(
            "estimated",
            model_name,
            str(config.get("currency", "USD")),
            value,
            str(config.get("pricing_version", "unspecified")),
        )

