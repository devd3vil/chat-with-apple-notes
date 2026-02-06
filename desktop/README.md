# Notes RAG Desktop (Milestone 1)

This is a minimal Tauri shell that starts the backend and shows `/health`.

## Dev Run

From repo root:

```bash
cd desktop/src-tauri
cargo tauri dev
```

Environment overrides:

- `BACKEND_CMD` (default: `python`)
- `BACKEND_ARGS` (extra args appended)
- `BACKEND_PORT` (default: `8001`)
- `BACKEND_WORKDIR` (default: repo root if detected)

The UI will poll `http://127.0.0.1:8001/health`.

## Notes

- The backend process is spawned on app startup and killed on window close.
- Next milestone adds a setup wizard and main tabs.
