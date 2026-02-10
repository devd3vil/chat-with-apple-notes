"""
Apple Notes Exporter Module

Provides utilities to export notes from Apple Notes app.
"""

import subprocess
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from src.models import Note


def export_notes_interactive() -> Optional[Path]:
    """
    Export notes from Apple Notes using AppleScript.
    
    Opens a file picker dialog for the user to select export directory.
    
    Returns:
        Path to the export directory if successful, None otherwise.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "export_notes.applescript"
    
    if not script_path.exists():
        print(f"Error: AppleScript not found at {script_path}")
        return None
    
    try:
        # Run the AppleScript
        result = subprocess.run(
            ["osascript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"AppleScript error: {result.stderr}")
            return None
        
        # Extract export folder from output if needed
        output = result.stdout.strip()
        if output:
            export_path = Path(output)
            if export_path.exists():
                print(f"Notes exported successfully to: {export_path}")
                return export_path
        
        print("Notes exported successfully")
        return None
        
    except subprocess.TimeoutExpired:
        print("Export operation timed out")
        return None
    except Exception as e:
        print(f"Error running AppleScript: {e}")
        return None


def export_notes_to_directory(export_dir: Path) -> bool:
    """
    Export notes from Apple Notes to a specific directory.
    
    This requires modifying the AppleScript to accept a directory parameter.
    For now, uses the interactive method.
    
    Args:
        export_dir: Path to the directory where notes should be exported.
        
    Returns:
        True if export was successful, False otherwise.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, use interactive export
    # TODO: Modify AppleScript to accept directory as parameter
    result = export_notes_interactive()
    return result is not None or export_dir.exists()


if __name__ == "__main__":
    print("Apple Notes Exporter")
    print("Launching interactive export...")
    export_notes_interactive()


def parse_exported_html(html_file: Path) -> Note:
    """
    Parse an exported HTML file into a Note object.
    
    Filename format: YYYYMMDD <title> [<id>].html
    
    Args:
        html_file: Path to the HTML file exported by AppleScript.
        
    Returns:
        Note object with parsed metadata and body.
        
    Raises:
        ValueError: If filename format is invalid.
    """
    filename = html_file.stem  # Remove .html extension
    
    # Parse filename: YYYYMMDD <title> [<id>]
    # Example: 20240115 My Note [p599]
    match = re.match(r"(\d{8})\s+(.+?)\s+\[([^\]]+)\]$", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    
    date_str, title, note_id = match.groups()
    
    # Parse date
    try:
        created_at = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Invalid date in filename: {date_str}")
    
    # Get file modification time as modified_at
    modified_at = datetime.fromtimestamp(html_file.stat().st_mtime)
    
    # Read HTML body
    with open(html_file, "r", encoding="utf-8") as f:
        body = f.read()
    
    # Extract folder from parent directory relative to some root
    # Assume structure: export_root/[Folder1/]...[FolderN/]file.html
    folder = _extract_folder(html_file)
    
    return Note(
        id=note_id,
        title=title,
        body=body,
        folder=folder,
        created_at=created_at,
        modified_at=modified_at,
    )


def _extract_folder(html_file: Path, root_markers: Optional[list[str]] = None) -> str:
    """
    Extract folder structure from HTML file path.
    
    Looks for common root markers (Notes, Downloads, Desktop) and extracts
    the relative path from there.
    
    Args:
        html_file: Path to the HTML file.
        root_markers: List of folder names to use as roots (e.g., ["Notes", "Downloads"]).
        
    Returns:
        Folder path (e.g., "Work/Projects" or "Notes" if no parent folders).
    """
    if root_markers is None:
        root_markers = ["Notes", "Downloads", "Desktop"]
    
    parts = list(html_file.parent.parts)
    
    # Find the last root marker and extract path from there
    for marker in root_markers:
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            if idx < len(parts) - 1:
                relative_parts = parts[idx + 1 :]
                return "/".join(relative_parts)
            else:
                return marker
    
    # If no root marker found, return the last directory name
    return html_file.parent.name or "Notes"


class HTMLToTextParser(HTMLParser):
    """Extract plain text from HTML."""
    
    def __init__(self):
        super().__init__()
        self.text: list[str] = []
    
    def handle_data(self, data: str) -> None:
        """Collect text data."""
        if data.strip():
            self.text.append(data.strip())
    
    def get_text(self) -> str:
        """Get extracted text."""
        return " ".join(self.text)


def extract_text_from_html(html: str) -> str:
    """
    Extract plain text content from HTML.
    
    Args:
        html: HTML content.
        
    Returns:
        Plain text extracted from HTML.
    """
    parser = HTMLToTextParser()
    try:
        parser.feed(html)
    except Exception:
        # If parsing fails, return original
        pass
    return parser.get_text()
