"""
Tests for incremental sync logic.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
import pytest
from src.models import Note
from src.sync import SyncState, has_note_changed, incremental_sync, _compute_hash


class TestSyncState:
    """Tests for SyncState class."""
    
    def test_sync_state_initialization(self):
        """Test creating a new SyncState."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "sync_state.json"
            state = SyncState(state_file)
            
            assert state.last_sync_time is None
            assert state.note_metadata == {}
    
    def test_sync_state_save_and_load(self):
        """Test persisting and loading sync state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "sync_state.json"
            
            # Create and save state
            state1 = SyncState(state_file)
            note1 = Note(
                id="p123",
                title="Test",
                body="Test body",
                created_at=datetime(2024, 1, 1),
                modified_at=datetime(2024, 1, 1),
            )
            state1.update_note(note1)
            state1.mark_sync_complete()
            
            # Load in new instance
            state2 = SyncState(state_file)
            assert state2.has_note("p123")
            assert state2.last_sync_time is not None
    
    def test_sync_state_has_note(self):
        """Test checking if note is tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SyncState(Path(tmpdir) / "state.json")
            
            note = Note(
                id="p123",
                title="Test",
                body="Body",
                created_at=datetime.now(),
                modified_at=datetime.now(),
            )
            state.update_note(note)
            
            assert state.has_note("p123")
            assert not state.has_note("p456")
    
    def test_sync_state_update_note(self):
        """Test updating note metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SyncState(Path(tmpdir) / "state.json")
            
            note = Note(
                id="p123",
                title="Test",
                body="Body content",
                created_at=datetime(2024, 1, 1),
                modified_at=datetime(2024, 1, 1),
            )
            state.update_note(note)
            
            stored_hash = state.get_note_hash("p123")
            assert stored_hash is not None
            assert stored_hash == _compute_hash("Body content")
    
    def test_sync_state_get_notes_to_remove(self):
        """Test detecting removed notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SyncState(Path(tmpdir) / "state.json")
            
            # Add notes to state
            state.note_metadata["p123"] = {"modified_at": "2024-01-01", "body_hash": "abc"}
            state.note_metadata["p456"] = {"modified_at": "2024-01-01", "body_hash": "def"}
            state.note_metadata["p789"] = {"modified_at": "2024-01-01", "body_hash": "ghi"}
            
            # Only p123 and p456 still exist
            removed = state.get_notes_to_remove({"p123", "p456"})
            assert set(removed) == {"p789"}
    
    def test_sync_state_corrupted_file(self):
        """Test handling corrupted state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text("{ invalid json")
            
            # Should not crash
            state = SyncState(state_file)
            assert state.last_sync_time is None
            assert state.note_metadata == {}


class TestComputeHash:
    """Tests for hash computation."""
    
    def test_compute_hash_deterministic(self):
        """Test that hash is deterministic."""
        text = "Hello, World!"
        hash1 = _compute_hash(text)
        hash2 = _compute_hash(text)
        assert hash1 == hash2
    
    def test_compute_hash_different_for_different_text(self):
        """Test that different text produces different hashes."""
        hash1 = _compute_hash("Text 1")
        hash2 = _compute_hash("Text 2")
        assert hash1 != hash2
    
    def test_compute_hash_empty_string(self):
        """Test hashing empty string."""
        hash_empty = _compute_hash("")
        assert len(hash_empty) == 64  # SHA256 produces 64 hex chars


class TestHasNoteChanged:
    """Tests for change detection."""
    
    def test_new_note_is_changed(self):
        """Test that new notes (no stored hash) are marked as changed."""
        note = Note(
            id="p123",
            title="New Note",
            body="New body",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        assert has_note_changed(note, None) is True
    
    def test_unchanged_note_not_changed(self):
        """Test that unchanged notes are not marked as changed."""
        body = "Unchanged body"
        note = Note(
            id="p123",
            title="Note",
            body=body,
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        stored_hash = _compute_hash(body)
        assert has_note_changed(note, stored_hash) is False
    
    def test_modified_note_is_changed(self):
        """Test that modified notes are marked as changed."""
        note = Note(
            id="p123",
            title="Note",
            body="New body content",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        old_hash = _compute_hash("Old body content")
        assert has_note_changed(note, old_hash) is True


class TestIncrementalSync:
    """Integration tests for incremental sync."""
    
    def test_incremental_sync_new_notes(self):
        """Test detecting new notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Create exported HTML files
            note1_file = export_dir / "20240115 Note 1 [p123].html"
            note1_file.write_text("<html>Content 1</html>")
            
            note2_file = export_dir / "20240116 Note 2 [p456].html"
            note2_file.write_text("<html>Content 2</html>")
            
            # First sync
            state = SyncState(state_file)
            changed, removed = incremental_sync(export_dir, state)
            
            assert len(changed) == 2
            assert len(removed) == 0
            assert {n.id for n in changed} == {"p123", "p456"}
    
    def test_incremental_sync_unchanged_notes(self):
        """Test that unchanged notes are not re-synced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Create exported note
            note_file = export_dir / "20240115 Note 1 [p123].html"
            note_file.write_text("<html>Content</html>")
            
            # First sync
            state = SyncState(state_file)
            changed1, _ = incremental_sync(export_dir, state)
            assert len(changed1) == 1
            
            # Second sync without changes
            changed2, _ = incremental_sync(export_dir, state)
            assert len(changed2) == 0  # No changes
    
    def test_incremental_sync_modified_note(self):
        """Test detecting modified notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Create and export note
            note_file = export_dir / "20240115 Note 1 [p123].html"
            note_file.write_text("<html>Original content</html>")
            
            # First sync
            state = SyncState(state_file)
            changed1, _ = incremental_sync(export_dir, state)
            assert len(changed1) == 1
            
            # Modify the note
            note_file.write_text("<html>Modified content</html>")
            
            # Second sync should detect change
            changed2, _ = incremental_sync(export_dir, state)
            assert len(changed2) == 1
            assert changed2[0].id == "p123"
    
    def test_incremental_sync_removed_note(self):
        """Test detecting removed notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Create two notes
            note1_file = export_dir / "20240115 Note 1 [p123].html"
            note1_file.write_text("<html>Content 1</html>")
            note2_file = export_dir / "20240116 Note 2 [p456].html"
            note2_file.write_text("<html>Content 2</html>")
            
            # First sync
            state = SyncState(state_file)
            incremental_sync(export_dir, state)
            
            # Remove one note
            note1_file.unlink()
            
            # Second sync should detect removal
            changed, removed = incremental_sync(export_dir, state)
            assert len(changed) == 0  # Only p456 remains, unchanged
            assert "p123" in removed
    
    def test_incremental_sync_nested_folders(self):
        """Test handling nested folder structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Create nested structure
            work_dir = export_dir / "Work" / "Projects"
            work_dir.mkdir(parents=True)
            
            note_file = work_dir / "20240115 Project Note [p123].html"
            note_file.write_text("<html>Content</html>")
            
            state = SyncState(state_file)
            changed, _ = incremental_sync(export_dir, state)
            
            assert len(changed) == 1
            assert changed[0].folder in ["Projects", "Work/Projects", "work_dir"]


class TestSyncStateIntegration:
    """Integration tests with real file operations."""
    
    def test_full_sync_cycle(self):
        """Test complete sync cycle: new → unchanged → modified → removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            
            # Cycle 1: Add two notes
            (export_dir / "20240115 Note A [pA].html").write_text("<html>A</html>")
            (export_dir / "20240116 Note B [pB].html").write_text("<html>B</html>")
            
            state = SyncState(state_file)
            changed1, removed1 = incremental_sync(export_dir, state)
            assert len(changed1) == 2
            assert len(removed1) == 0
            
            # Cycle 2: Modify A, B unchanged
            (export_dir / "20240115 Note A [pA].html").write_text("<html>A modified</html>")
            changed2, removed2 = incremental_sync(export_dir, state)
            assert len(changed2) == 1
            assert changed2[0].id == "pA"
            assert len(removed2) == 0
            
            # Cycle 3: Remove B
            (export_dir / "20240116 Note B [pB].html").unlink()
            changed3, removed3 = incremental_sync(export_dir, state)
            assert len(changed3) == 0  # A unchanged
            assert "pB" in removed3
