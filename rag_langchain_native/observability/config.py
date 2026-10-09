"""Observability 独立配置；默认本地、默认不采集业务内容。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_SETTINGS


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObservabilitySettings:
    """Phase 10 配置。

    ``capture_content=False`` 是重要的隐私默认值：问题、Prompt、Context、Answer 与文档正文
    不进入 Trace/Log。``exporter=local`` 只写 V3 runtime；只有显式设置 ``otlp`` 才联网。
    """

    enabled: bool = _bool("V3_OBSERVABILITY_ENABLED", True)
    exporter: str = os.getenv("V3_OBSERVABILITY_EXPORTER", "local").lower()
    sample_rate: float = float(os.getenv("V3_OBSERVABILITY_SAMPLE_RATE", "1.0"))
    log_level: str = os.getenv("V3_OBSERVABILITY_LOG_LEVEL", "INFO").upper()
    capture_content: bool = _bool("V3_OBSERVABILITY_CAPTURE_CONTENT", False)
    otlp_endpoint: str | None = os.getenv("V3_OBSERVABILITY_OTLP_ENDPOINT")
    retention_days: int = int(os.getenv("V3_OBSERVABILITY_RETENTION_DAYS", "7"))
    max_file_bytes: int = int(os.getenv("V3_OBSERVABILITY_MAX_FILE_BYTES", "5242880"))
    backup_count: int = int(os.getenv("V3_OBSERVABILITY_BACKUP_COUNT", "5"))
    runtime_dir: Path = DEFAULT_SETTINGS.runtime_dir / "observability"
    service_name: str = "rag-langchain-native"
    pricing_path: Path = DEFAULT_SETTINGS.runtime_dir / "observability" / "pricing.json"

    def validate(self) -> None:
        if self.exporter not in {"local", "otlp", "none"}:
            raise ValueError("V3_OBSERVABILITY_EXPORTER must be local, otlp, or none")
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("V3_OBSERVABILITY_SAMPLE_RATE must be between 0 and 1")
        if self.exporter == "otlp" and not self.otlp_endpoint:
            raise ValueError("OTLP exporter requires V3_OBSERVABILITY_OTLP_ENDPOINT")
        if self.retention_days < 1 or self.max_file_bytes < 1024 or self.backup_count < 1:
            raise ValueError("Observability retention/rotation values are invalid")

    def ensure_directory(self) -> None:
        self.validate()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_OBSERVABILITY_SETTINGS = ObservabilitySettings()

