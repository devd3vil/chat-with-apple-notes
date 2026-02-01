"""
Incremental sync logic for Apple Notes.

Tracks note metadata and detects changes to avoid re-processing unchanged notes.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.models import Note


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
        self.last_sync_time: Optional[datetime] = None
        self.note_metadata: dict[str, dict] = {}  # note_id -> {modified_at, body_hash}
        
        self._load()
    
    def _load(self) -> None:
        """Load sync state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.last_sync_time = (
                        datetime.fromisoformat(data["last_sync_time"])
                        if data.get("last_sync_time")
                        else None
                    )
                    self.note_metadata = data.get("note_metadata", {})
            except (json.JSONDecodeError, KeyError):
                # If file is corrupted, start fresh
                self.last_sync_time = None
                self.note_metadata = {}
    
    def save(self) -> None:
        """Persist sync state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "last_sync_time": (
                        self.last_sync_time.isoformat()
                        if self.last_sync_time
                        else None
                    ),
                    "note_metadata": self.note_metadata,
                },
                f,
                indent=2,
            )
    
    def has_note(self, note_id: str) -> bool:
        """Check if note is tracked."""
        return note_id in self.note_metadata
    
    def get_note_hash(self, note_id: str) -> Optional[str]:
        """Get stored hash for a note, or None if not tracked."""
        return self.note_metadata.get(note_id, {}).get("body_hash")
    
    def update_note(self, note: Note) -> None:
        """Update tracked state for a note."""
        body_hash = _compute_hash(note.body)
        self.note_metadata[note.id] = {
            "modified_at": note.modified_at.isoformat(),
            "body_hash": body_hash,
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
    
    from src.notes_exporter import parse_exported_html
    
    # Collect all exported notes
    html_files = list(export_dir.rglob("*.html"))
    exported_notes: dict[str, Note] = {}
    
    for html_file in html_files:
        try:
            note = parse_exported_html(html_file)
            exported_notes[note.id] = note
        except Exception as e:
            print(f"Warning: Failed to parse {html_file}: {e}")
            continue
    
    # Detect changes
    changed_notes: list[Note] = []
    
    for note_id, note in exported_notes.items():
        stored_hash = sync_state.get_note_hash(note_id)
        
        if has_note_changed(note, stored_hash):
            changed_notes.append(note)
            sync_state.update_note(note)
    
    # Detect removals
    removed_note_ids = sync_state.get_notes_to_remove(set(exported_notes.keys()))
    for note_id in removed_note_ids:
        del sync_state.note_metadata[note_id]
    
    # Save updated state
    sync_state.mark_sync_complete()
    
    return changed_notes, removed_note_ids
