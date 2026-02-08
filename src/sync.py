"""
Incremental sync logic for Apple Notes.

Tracks note metadata and detects changes to avoid re-processing unchanged notes.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.models import Note

SYNC_STATE_SCHEMA_VERSION = 2


class SyncState:
    """
    Tracks sync state to enable incremental syncing.
    
    Persists note metadata (id, modification_date, body hash) to disk.
    """
    
    def __init__(self, state_file: Path = Path("data/sync_state.json")):
        """
        Initialize sync state from disk or create new.
        
        Args:
            state_file: Path to persist sync state JSON file.
        """
        self.state_file = state_file
        self.schema_version = SYNC_STATE_SCHEMA_VERSION
        self.last_sync_time: Optional[datetime] = None
        # note_id -> metadata:
        # {
        #   source_id, source_path, source_mtime, content_hash, indexed_at, schema_version
        # }
        self.note_metadata: dict[str, dict] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Load sync state from disk if it exists."""
        needs_resave = False
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_schema_version = int(data.get("schema_version") or 1)
                    self.last_sync_time = (
                        datetime.fromisoformat(data["last_sync_time"])
                        if data.get("last_sync_time")
                        else None
                    )
                    raw_note_metadata = data.get("note_metadata", {})
                    self.note_metadata = self._normalize_note_metadata(
                        raw_note_metadata,
                        file_schema_version,
                    )
                    if file_schema_version != self.schema_version:
                        needs_resave = True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # If file is corrupted, start fresh
                self.last_sync_time = None
                self.note_metadata = {}
        if needs_resave:
            self.save()

    def _normalize_note_metadata(
        self,
        raw_note_metadata: dict,
        file_schema_version: int,
    ) -> dict[str, dict]:
        """Normalize old/new manifest data into current schema."""
        normalized: dict[str, dict] = {}
        indexed_at_fallback = (
            self.last_sync_time.isoformat() if self.last_sync_time else datetime.now().isoformat()
        )
        for note_id, meta in raw_note_metadata.items():
            meta = meta or {}
            content_hash = meta.get("content_hash") or meta.get("body_hash") or ""
            source_mtime = meta.get("source_mtime")
            if source_mtime is None:
                source_mtime = meta.get("modified_at")
            normalized[note_id] = {
                "source_id": meta.get("source_id") or note_id,
                "source_path": meta.get("source_path"),
                "source_mtime": source_mtime,
                "content_hash": content_hash,
                "indexed_at": meta.get("indexed_at") or indexed_at_fallback,
                "schema_version": int(meta.get("schema_version") or self.schema_version),
                # Legacy fields retained for backward compatibility.
                "modified_at": meta.get("modified_at"),
                "body_hash": content_hash,
            }

        # Upgrade path for manifest files from pre-schema versions.
        if file_schema_version < self.schema_version:
            for note_meta in normalized.values():
                note_meta["schema_version"] = self.schema_version
        return normalized
    
    def save(self) -> None:
        """Persist sync state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": self.schema_version,
            "last_sync_time": (
                self.last_sync_time.isoformat()
                if self.last_sync_time
                else None
            ),
            "note_metadata": self.note_metadata,
        }
        backup_file = self.state_file.with_suffix(f"{self.state_file.suffix}.bak")
        tmp_file = self.state_file.with_suffix(f"{self.state_file.suffix}.tmp")

        if self.state_file.exists():
            shutil.copy2(self.state_file, backup_file)

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.state_file)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            if backup_file.exists() and not self.state_file.exists():
                shutil.copy2(backup_file, self.state_file)
            raise
    
    def has_note(self, note_id: str) -> bool:
        """Check if note is tracked."""
        return note_id in self.note_metadata
    
    def get_note_hash(self, note_id: str) -> Optional[str]:
        """Get stored hash for a note, or None if not tracked."""
        metadata = self.note_metadata.get(note_id, {})
        return metadata.get("content_hash") or metadata.get("body_hash")
    
    def update_note(
        self,
        note: Note,
        source_path: Optional[str] = None,
        source_mtime: Optional[float] = None,
    ) -> None:
        """Update tracked manifest state for a note."""
        content_hash = _compute_hash(note.body)
        indexed_at = datetime.now().isoformat()
        self.note_metadata[note.id] = {
            "source_id": note.id,
            "source_path": source_path,
            "source_mtime": source_mtime,
            "content_hash": content_hash,
            "indexed_at": indexed_at,
            "schema_version": self.schema_version,
            # Legacy fields retained for backward compatibility.
            "modified_at": note.modified_at.isoformat(),
            "body_hash": content_hash,
        }
    
    def get_notes_to_remove(self, current_note_ids: set[str]) -> list[str]:
        """
        Get note IDs that were removed from the export.
        
        Args:
            current_note_ids: Set of note IDs currently in export.
            
        Returns:
            List of note IDs that are no longer in the export.
        """
        return [nid for nid in self.note_metadata.keys() if nid not in current_note_ids]
    
    def mark_sync_complete(self) -> None:
        """Mark sync as complete and save state."""
        self.last_sync_time = datetime.now()
        self.save()


def _compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_note_changed(note: Note, stored_hash: Optional[str]) -> bool:
    """
    Check if a note has been modified since last sync.
    
    Args:
        note: The note to check.
        stored_hash: The previously stored body hash, or None if never synced.
        
    Returns:
        True if note is new or modified, False if unchanged.
    """
    if stored_hash is None:
        return True  # New note
    
    current_hash = _compute_hash(note.body)
    return current_hash != stored_hash


def incremental_sync(
    export_dir: Path,
    sync_state: Optional[SyncState] = None,
) -> tuple[list[Note], list[str]]:
    """
    Perform incremental sync of notes from export directory.
    
    Args:
        export_dir: Directory where AppleScript exported notes.
        sync_state: Optional SyncState to use; creates new if not provided.
        
    Returns:
        Tuple of (changed_notes, removed_note_ids):
        - changed_notes: List of Notes that are new or modified.
        - removed_note_ids: List of note IDs that were deleted.
    """
    if sync_state is None:
        sync_state = SyncState()
    
    changed, removed, _summary = incremental_sync_detailed(export_dir, sync_state)
    return changed, removed


def incremental_sync_detailed(
    export_dir: Path,
    sync_state: Optional[SyncState] = None,
) -> tuple[list[Note], list[str], dict[str, int]]:
    """Perform incremental sync and return a delta summary."""
    if sync_state is None:
        sync_state = SyncState()

    from src.notes_exporter import parse_exported_html

    # Collect all exported notes
    html_files = list(export_dir.rglob("*.html"))
    exported_notes: dict[str, Note] = {}
    note_sources: dict[str, tuple[str, float]] = {}

    for html_file in html_files:
        try:
            note = parse_exported_html(html_file)
            exported_notes[note.id] = note
            note_sources[note.id] = (str(html_file), html_file.stat().st_mtime)
        except Exception as e:
            print(f"Warning: Failed to parse {html_file}: {e}")
            continue

    # Detect changes
    changed_notes: list[Note] = []
    added_notes = 0
    updated_notes = 0
    unchanged_notes = 0

    for note_id, note in exported_notes.items():
        was_tracked = sync_state.has_note(note_id)
        stored_hash = sync_state.get_note_hash(note_id)

        if has_note_changed(note, stored_hash):
            changed_notes.append(note)
            source_path, source_mtime = note_sources.get(note_id, (None, None))
            sync_state.update_note(
                note,
                source_path=source_path,
                source_mtime=source_mtime,
            )
            if was_tracked:
                updated_notes += 1
            else:
                added_notes += 1
        else:
            unchanged_notes += 1

    # Detect removals
    removed_note_ids = sync_state.get_notes_to_remove(set(exported_notes.keys()))
    for note_id in removed_note_ids:
        del sync_state.note_metadata[note_id]

    # Save updated state
    sync_state.mark_sync_complete()

    summary = {
        "scanned_notes": len(exported_notes),
        "added_notes": added_notes,
        "updated_notes": updated_notes,
        "unchanged_notes": unchanged_notes,
        "removed_notes": len(removed_note_ids),
    }
    return changed_notes, removed_note_ids, summary
