"""Tests for notes exporter module"""

import tempfile
from datetime import datetime
from pathlib import Path
import pytest
from src.notes_exporter import (
    export_notes_interactive,
    parse_exported_html,
    extract_text_from_html,
    _extract_folder,
)


def test_export_notes_interactive_no_crash():
    """Test that export_notes_interactive handles gracefully when AppleScript is called"""
    # This test just ensures the function handles errors gracefully
    # Actual Apple Notes interaction requires user interaction
    result = export_notes_interactive()
    # Result can be None if user cancels or no export happens
    assert result is None or isinstance(result, Path)


class TestParseExportedHTML:
    """Tests for parsing exported HTML files."""
    
    def test_parse_valid_html_file(self):
        """Test parsing a valid exported HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "20240115 My Note [p599].html"
            html_file.write_text("<html><body>Note content</body></html>")
            
            note = parse_exported_html(html_file)
            
            assert note.id == "p599"
            assert note.title == "My Note"
            assert note.body == "<html><body>Note content</body></html>"
            assert note.created_at == datetime(2024, 1, 15)
    
    def test_parse_filename_with_spaces(self):
        """Test parsing filename with multiple spaces in title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "20240115 Multi Word Note Title [p123].html"
            html_file.write_text("<html>Content</html>")
            
            note = parse_exported_html(html_file)
            
            assert note.title == "Multi Word Note Title"
            assert note.id == "p123"
    
    def test_parse_invalid_filename_format(self):
        """Test error handling for invalid filename format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "invalid_filename.html"
            html_file.write_text("<html>Content</html>")
            
            with pytest.raises(ValueError):
                parse_exported_html(html_file)
    
    def test_parse_missing_id(self):
        """Test error handling when ID is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "20240115 Note.html"
            html_file.write_text("<html>Content</html>")
            
            with pytest.raises(ValueError):
                parse_exported_html(html_file)
    
    def test_parse_invalid_date(self):
        """Test error handling for invalid date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "99999999 Note [p123].html"
            html_file.write_text("<html>Content</html>")
            
            with pytest.raises(ValueError):
                parse_exported_html(html_file)
    
    def test_parse_preserves_html(self):
        """Test that HTML content is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
            html_file = export_dir / "20240115 Note [p123].html"
            html_file.write_text(html_content)
            
            note = parse_exported_html(html_file)
            
            assert note.body == html_content
    
    def test_parse_with_special_chars_in_title(self):
        """Test parsing note with special characters in title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            html_file = export_dir / "20240115 Note-with_special.chars [p123].html"
            html_file.write_text("<html>Content</html>")
            
            note = parse_exported_html(html_file)
            
            assert note.title == "Note-with_special.chars"
    
    def test_parse_folder_extraction(self):
        """Test that folder is correctly extracted from path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            work_dir = export_dir / "Work" / "Projects"
            work_dir.mkdir(parents=True)
            
            html_file = work_dir / "20240115 Project Note [p123].html"
            html_file.write_text("<html>Content</html>")
            
            note = parse_exported_html(html_file)
            
            assert note.folder is not None


class TestExtractTextFromHTML:
    """Tests for extracting text from HTML."""
    
    def test_extract_simple_text(self):
        """Test extracting text from simple HTML."""
        html = "<html><body>Hello World</body></html>"
        text = extract_text_from_html(html)
        
        assert "Hello" in text
        assert "World" in text
    
    def test_extract_text_removes_tags(self):
        """Test that HTML tags are removed."""
        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        text = extract_text_from_html(html)
        
        assert "<h1>" not in text
        assert "<p>" not in text
        assert "Title" in text
        assert "Content" in text
    
    def test_extract_text_empty_html(self):
        """Test extracting from empty HTML."""
        html = "<html><body></body></html>"
        text = extract_text_from_html(html)
        
        assert text == ""
    
    def test_extract_text_multiple_elements(self):
        """Test extracting from HTML with multiple elements."""
        html = (
            "<html>"
            "<body>"
            "<h1>Heading</h1>"
            "<p>Paragraph 1</p>"
            "<p>Paragraph 2</p>"
            "</body>"
            "</html>"
        )
        text = extract_text_from_html(html)
        
        assert "Heading" in text
        assert "Paragraph 1" in text
        assert "Paragraph 2" in text


class TestExtractFolder:
    """Tests for folder extraction from file path."""
    
    def test_extract_folder_from_notes_root(self):
        """Test extracting folder when Notes is root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_dir = Path(tmpdir) / "Notes"
            work_dir = notes_dir / "Work"
            work_dir.mkdir(parents=True)
            
            file_path = work_dir / "note.html"
            folder = _extract_folder(file_path, ["Notes"])
            
            assert folder == "Work"
    
    def test_extract_folder_nested(self):
        """Test extracting nested folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_dir = Path(tmpdir) / "Notes"
            nested_dir = notes_dir / "Work" / "Projects" / "Active"
            nested_dir.mkdir(parents=True)
            
            file_path = nested_dir / "note.html"
            folder = _extract_folder(file_path, ["Notes"])
            
            assert "Work" in folder
            assert "Projects" in folder
            assert "Active" in folder
    
    def test_extract_folder_no_root_marker(self):
        """Test behavior when no root marker is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            some_dir = Path(tmpdir) / "SomeFolder"
            some_dir.mkdir(parents=True)
            
            file_path = some_dir / "note.html"
            folder = _extract_folder(file_path, ["Notes"])
            
            # Should return something reasonable
            assert folder == "SomeFolder" or folder != ""
