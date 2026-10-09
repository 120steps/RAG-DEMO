"""V3 一致性备份与安全恢复。

SQLite 使用官方 backup API；文件和 Chroma 在同一进程停写锁内快照。该锁只覆盖单进程，
生产多副本必须先进入维护模式并停止所有 Writer，不能把本实现误称为分布式事务。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings
from .runtime_lock import RUNTIME_WRITE_LOCK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BackupService:
    """创建带哈希 Manifest 的快照，并只恢复到新的空目录。"""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def create(self, destination: str | Path) -> Path:
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError("Backup destination already exists")
        target.mkdir(parents=True)
        backup_id = uuid.uuid4().hex
        with RUNTIME_WRITE_LOCK:
            catalog_target = target / "catalog.sqlite3"
            source = sqlite3.connect(self.settings.catalog_path)
            output = sqlite3.connect(catalog_target)
            try:
                source.backup(output)
            finally:
                output.close()
                source.close()
            for source_dir, name in (
                (self.settings.upload_dir, "uploads"),
                (self.settings.chroma_dir, "chroma"),
            ):
                if source_dir.exists():
                    shutil.copytree(source_dir, target / name)
        files = {
            str(path.relative_to(target)).replace("\\", "/"): _sha256(path)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "backup_id": backup_id,
            "created_at": datetime.now(UTC).isoformat(),
            "format_version": 1,
            "files": files,
            "consistency": "single-process write lock; multi-process writers must be stopped",
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @staticmethod
    def verify(backup_dir: str | Path) -> dict:
        root = Path(backup_dir).resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("files", {}).items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"Backup integrity verification failed: {name}")
        return manifest

    def restore(self, backup_dir: str | Path, destination: str | Path) -> Path:
        """恢复到全新目录；绝不覆盖当前 runtime。"""
        source = Path(backup_dir).resolve()
        target = Path(destination).resolve()
        self.verify(source)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("Restore destination must be empty")
        if target == self.settings.runtime_dir.resolve():
            raise ValueError("Refusing to overwrite the active runtime")
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name == "manifest.json":
                continue
            destination_item = target / item.name
            shutil.copytree(item, destination_item) if item.is_dir() else shutil.copy2(
                item, destination_item
            )
        shutil.copy2(source / "manifest.json", target / "backup_manifest.json")
        return target
