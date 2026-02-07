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
- `BACKEND_PORT` (optional fixed port; if unset, app picks an available local port starting from `8001`)
- `BACKEND_WORKDIR` (default: repo root if detected)

The UI resolves backend base URL from the app shell and polls `<base_url>/health`.

## Notes

- The backend process is spawned on app startup and killed on window close.
- Next milestone adds a setup wizard and main tabs.
