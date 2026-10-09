"""单进程滑动窗口限流；生产多副本应由网关或 Redis 提供全局限流。"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class LocalRateLimiter:
    """按客户端键限制一分钟内请求数，不保存密码、Token 或问题正文。"""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """返回是否允许以及需要等待的秒数；使用单调时钟避免系统时间回拨。"""
        current = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            cutoff = current - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False, max(1, int(self.window_seconds - (current - events[0])))
            events.append(current)
            return True, 0
