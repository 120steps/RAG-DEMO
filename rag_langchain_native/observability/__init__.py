"""Phase 10 可观测性公共入口。

业务模块只依赖这里暴露的 ``get_observability``，关闭功能时 Manager 返回低开销 no-op
上下文，不改变 RAG 算法或异常语义。
"""

from .config import ObservabilitySettings
from .core import (
    ObservabilityManager,
    configure_observability,
    get_observability,
)

__all__ = [
    "ObservabilityManager",
    "ObservabilitySettings",
    "configure_observability",
    "get_observability",
]

