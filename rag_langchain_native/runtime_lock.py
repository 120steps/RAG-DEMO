"""SQLite、上传目录与 Chroma 跨组件快照使用的进程内停写锁。"""

from __future__ import annotations

import threading
from functools import wraps


RUNTIME_WRITE_LOCK = threading.RLock()


def serialized_runtime_write(function):
    """序列化生命周期写操作；多进程部署必须改用外部锁/维护窗口。"""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with RUNTIME_WRITE_LOCK:
            return function(*args, **kwargs)

    return wrapped
