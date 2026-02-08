# Notes RAG Desktop (Milestone 1-3)

This Tauri shell:

- starts the backend sidecar and checks `/health`,
- includes a setup wizard panel for Ollama detection/start/install guidance,
- lets users choose models, pull them locally, and persist setup config.
- supports retrying failed pulls and resuming pending model pulls.
- includes Step 3 tabs for Search, Ask, and Sync/Health with folder ingest actions.
- surfaces folder permission-denied errors with retry guidance for macOS privacy settings.
- uses backend ingest modes (`full` / `delta`) and displays sync summary in the UI.

## Dev Run

From repo root:

```bash
cd desktop/src-tauri
cargo tauri dev
```

If your machine has Tauri CLI v2 globally, run with a v1 CLI compatible with this app:

```bash
PATH="$PWD/../../.tools/bin:$PATH" cargo tauri dev
```

Environment overrides:

- `BACKEND_CMD` (default: `python`)
- `BACKEND_ARGS` (extra args appended)
- `BACKEND_PORT` (optional fixed port; if unset, app picks an available local port starting from `8001`)
- `BACKEND_WORKDIR` (default: repo root if detected)

The UI resolves backend base URL from the app shell and polls `<base_url>/health`.

## Notes

- The backend process is spawned on app startup and killed on window close.
- Setup wizard config is stored in app config dir as `setup.json`.
- Folder picker uses a macOS native chooser via AppleScript.
