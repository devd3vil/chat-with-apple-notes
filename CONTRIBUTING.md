# Contributing to Apple Notes RAG Q&A

Thank you for your interest in contributing! This document provides guidelines and instructions.

## Code of Conduct

Be respectful and inclusive. All contributors are expected to maintain a welcoming environment.

## Getting Started

### Prerequisites

- Python 3.12+
- macOS (for Apple Notes integration features)
- Ollama (for testing LLM/embedding functionality)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/chatwithNotes.git
cd chatwithNotes

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Keep changes focused and atomic
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type check
mypy src/

# Run tests
pytest -v
```

All checks must pass before submitting a PR.

### 4. Commit with Clear Messages

```bash
git commit -m "feat: add new RAG feature"
# or
git commit -m "fix: handle edge case in retriever"
```

Use conventional commit types:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test additions/changes
- `refactor:` - Code restructuring
- `perf:` - Performance improvements
- `chore:` - Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear description of changes
- Reference to any related issues
- Test results/screenshots if applicable

## Testing Guidelines

### Test Structure

Tests are organized by module in the `tests/` directory:

- `test_models.py` - Data model validation
- `test_config.py` - Configuration loading
- `test_sync.py` - Note synchronization
- `test_embedder.py` - Embedding generation
- `test_store.py` - Vector store operations
- `test_retriever.py` - Retrieval logic
- `test_llm.py` - LLM interface
- `test_chain.py` - RAG chain orchestration

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_feature_does_something():
    """Test description."""
    # Arrange
    mock_dependency = Mock()

    # Act
    result = my_function(mock_dependency)

    # Assert
    assert result == expected_value
```

**Key principles:**

- Use fake implementations (`FakeEmbedder`, `FakeLLM`, `InMemoryStore`) for unit tests
- Mock external dependencies (Ollama, Chroma)
- Make tests deterministic (no flakiness)
- Aim for high coverage on critical paths

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_chain.py

# With coverage report
pytest --cov=src --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Code Style

### Python Style Guide

Follow PEP 8 and use type hints:

```python
def process_notes(notes: list[Note]) -> QAResult:
    """Process notes and return Q&A result.

    Args:
        notes: List of notes to process.

    Returns:
        QAResult with answer and citations.

    Raises:
        ValueError: If notes list is empty.
    """
    if not notes:
        raise ValueError("Cannot process empty notes list")

    # Implementation...
    return result
```

### Key Style Points

- Use type hints for all function arguments and returns
- Write docstrings for all public functions/classes
- Keep lines under 100 characters
- Use meaningful variable names
- One import per line (except `from ... import a, b`)

### Formatting Tools

```bash
# Auto-format with Black
black src/ tests/

# Check with Ruff
ruff check src/ tests/

# Type checking with Mypy
mypy src/
```

## Architecture & Design Decisions

### Core Principles

1. **Local-First**: No external APIs, all processing local
2. **Testability**: Deterministic fakes for unit tests
3. **Modularity**: Abstract base classes (ABC) for extensibility
4. **Type Safety**: Full type hints throughout
5. **Documentation**: Clear docstrings and inline comments

### Adding New Features

If adding a new component:

1. Define abstract base class (ABC) if it's pluggable
2. Implement production version (e.g., OllamaEmbedder)
3. Implement fake/test version (e.g., FakeEmbedder)
4. Add comprehensive unit tests with both implementations
5. Update documentation

Example structure for a new store type:

```python
# In src/store.py (add to abstract VectorStore)
class MyCustomStore(VectorStore):
    """My custom vector store implementation."""

    def __init__(self, config: Settings):
        self.config = config

    def add(self, chunk_id: str, text: str, embedding: list[float]) -> None:
        """Add chunk to store."""
        pass

    # Implement other abstract methods...
```

```python
# In tests/test_store.py (add new test class)
class TestMyCustomStore:
    def test_add_chunk(self):
        """Test adding chunk to custom store."""
        store = MyCustomStore(Settings())
        # Test implementation...
```

## Documentation

### What to Document

- **README.md**: User-facing setup and usage
- **Code comments**: Why decisions were made (not what code does)
- **Docstrings**: Public API documentation
- **CONTRIBUTING.md**: Developer guidelines (this file)

### Documentation Style

- Write in clear, active voice
- Use examples where helpful
- Keep documentation in sync with code
- Link to relevant sections/files

## Common Tasks

### Adding a New Configuration Option

1. Add to `src/config.py` Settings class
2. Add to `.env.example` with description
3. Update README configuration section
4. Add test in `tests/test_config.py`

### Adding a New Model Field

1. Add to `src/models.py` Pydantic model
2. Add validation if needed
3. Update tests in `tests/test_models.py`
4. Update any serialization/deserialization

### Fixing a Bug

1. Create a test case that reproduces the bug
2. Fix the code
3. Verify test passes
4. Add regression test to prevent future breakage

## Performance Considerations

- Profile code before optimizing
- Use FakeEmbedder for development (no Ollama latency)
- Batch operations when possible
- Consider memory usage with large note sets
- Use incremental sync to avoid re-processing

## Questions?

- Check existing issues and discussions
- Read the codebase (it's small and well-organized!)
- Ask in issues or discussions section

## Review Process

PRs will be reviewed for:

✅ Code quality (style, types, patterns)  
✅ Test coverage (new features need tests)  
✅ Documentation (README, docstrings updated)  
✅ Performance (no regressions)  
✅ Security (no exposed secrets, safe practices)

Feedback will be constructive and collaborative. Don't hesitate to ask questions!

---

**Thank you for contributing to making Apple Notes RAG better for everyone!** 🎉
