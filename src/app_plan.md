# plan.md — Apple Notes Local RAG macOS App

## Summary

Convert the existing local Apple Notes RAG backend (FastAPI + vector+BM25 + rerank/compact + citations) into an installable macOS desktop application that:

- guides the user through setup via a click-through wizard,
- installs/verifies Ollama and required local models,
- helps the user export Apple Notes into a local folder (and/or exports via Notes automation later),
- ingests notes into a local index,
- provides a main UI for Semantic Search, Q&A with citations, and Sync (delta reindex),
- runs fully locally (privacy-first), with health/stats screens.

The backend APIs are already working:

- POST `/ingest`
- POST `/ask`
- POST `/search`
- GET `/health`
- GET `/stats`

---

## Goals

1. **One-click install**: user can download and install on macOS.
2. **Click-through setup**: wizard verifies/installs Ollama, pulls models, selects notes folder, ingests, and finishes setup.
3. **Simple daily use**: main screen supports:
   - semantic search (retrieval-only),
   - Q&A (LLM + citations + confidence),
   - sync (delta ingestion),
   - health/stats.
4. **Local-only**: no cloud dependency by default (all data stays on device).

## Non-goals (v1)

- iCloud account sign-in
- cross-device sync
- mobile apps
- advanced “agent” workflows
- full fidelity rendering of all Apple Notes attachments (we’ll ingest text first; attachments later)

---

## Key Decisions

### App shell choice (pick one)

- **Option A (recommended): Tauri** (UI in web stack, native shell, lighter than Electron)
- **Option B: SwiftUI** (native UI, easiest for AppleScript integrations)

### Backend packaging

Goal: end users should NOT need Python/venv/pip.

- Build backend into a self-contained executable:
  - PyInstaller (fastest to implement) or Nuitka (often better performance).
- Bundle it inside the macOS app as a **sidecar** process.
- App is responsible for:
  - starting/stopping backend,
  - selecting an available localhost port,
  - verifying readiness via `GET /health`,
  - surfacing logs in a debug panel.

### Ollama install

The app should guide install; do not silently escalate privileges.

- Detect Ollama by:
  - checking `ollama` binary presence OR pinging `http://127.0.0.1:11434`.
- If missing, present guided steps:
  - open official installer (DMG) OR Homebrew path if available.
- Once installed, pull models via:
  - `ollama pull <embed_model>`
  - `ollama pull <chat_model>`

---

## User Experience

### First-run Setup Wizard

1. **Welcome**
   - Explain: runs locally; asks for folder access for Notes export.
2. **Backend check**
   - Start backend sidecar.
   - Confirm `GET /health` passes.
3. **Ollama check**
   - Detect Ollama running locally.
   - If not installed/running:
     - show “Install Ollama” (guided) + “Start Ollama” actions.
4. **Model download**
   - Pull required models.
   - Show progress + disk usage estimate.
5. **Notes source**
   - v1: “Choose a Notes Export Folder”
   - v2: “Export Notes from Notes.app (1 click)” (AppleScript/ScriptingBridge)
6. **Ingest**
   - Trigger `POST /ingest` with folder path.
   - Show progress and post-ingest summary from `/stats`.
7. **Done**
   - Launch main screen.

### Main Screen

Tabs:

1. **Semantic Search**
   - input → `POST /search`
   - list results with snippet + open file button
2. **Ask (Q&A)**
   - input → `POST /ask`
   - show answer + confidence + citations panel
3. **Sync**
   - show last sync time (`GET /stats`)
   - “Sync now” triggers delta ingest
4. **Health / Stats**
   - `GET /health` and `GET /stats`
   - show store size, doc counts, model names, last sync

---

## Backend Changes (Minimal, App-friendly)

### Add ingestion modes

Update `/ingest` to support:

- `mode = "full"` (reindex everything)
- `mode = "delta"` (only changed files/notes)

### Add/ensure manifest storage for delta sync

Store a small manifest (sqlite or JSON in app data dir) keyed by source document:

- `source_id` (file path or note id)
- `mtime`
- `content_hash` (sha256 of normalized text)
- `indexed_at`

Delta logic:

- if new file → ingest
- if hash changed → reingest
- if deleted file → delete vectors + bm25 entries

### Ensure app-data directories

Use OS-appropriate paths (macOS Application Support):

- index db (Chroma)
- bm25 artifacts
- manifest
- logs

### Improve `/stats`

Return:

- total docs/notes
- chunk count
- last ingest time
- delta counts (added/updated/deleted) for last run
- configured models
- store size

---

## App Responsibilities

### Local service manager

- pick port (or use fixed port and detect conflicts)
- spawn backend sidecar with env vars:
  - `APP_DATA_DIR`
  - `NOTES_EXPORT_DIR`
  - `OLLAMA_HOST` (optional)
  - `EMBED_MODEL`, `CHAT_MODEL`
- health-check loop until ready

### Ollama/model manager

- detect Ollama running
- if missing, guide install
- if running, ensure models are available:
  - run `ollama list` (or pull and handle “already exists”)
- show progress & errors

### Folder access + UX

- folder picker
- store selected export folder
- validate folder contents before ingest (non-empty; supported file types)

---

## Notes Export Approach

### v1 (recommended for speed): Folder-based ingestion

- App guides user to export notes to a local folder (manual step).
- Ingest reads files from that folder.

### v2 (optional, big UX win): One-click export from Notes.app

- Use AppleScript/ScriptingBridge to read notes and write:
  - one file per note (html/markdown)
  - include title, created/modified timestamps
- Then run delta ingest.

---

## UI Requirements

### Semantic Search

- query input
- results list:
  - title/source
  - snippet
  - score
  - open source (Finder reveal / open file)
  - “Ask about this” shortcut (prefill Q&A)

### Ask (Q&A)

- question input
- answer view
- confidence indicator (0–1)
- citations list:
  - source + snippet
  - open source action

### Sync

- “Sync now”
- progress indicator
- results summary: added/updated/deleted

### Health/Stats

- backend health
- ollama health
- model names
- store stats

---

## Packaging & Distribution

### Build artifacts

- macOS `.app` packaged into `.dmg`
- backend bundled as sidecar executable
- config stored in app data dir

### Signing & notarization (optional but recommended)

- prepare for codesigning + notarization workflow
- provide unsigned builds for dev/testing

---

## Milestones

### Milestone 1 — App shell + backend sidecar

**Deliverable**

- App launches
- backend starts
- `/health` displayed in UI

**Acceptance**

- User can open app and see “Backend: Healthy”

### Milestone 2 — Setup Wizard: Ollama + model download

**Deliverable**

- Detect/install/start Ollama (guided)
- Pull selected models
- Persist config

**Acceptance**

- Wizard completes even on a clean Mac (with user clicking through)

### Milestone 3 — Folder selection + ingest + main features

**Deliverable**

- Pick export folder
- run `/ingest`
- Semantic Search and Ask tabs work end-to-end

**Acceptance**

- Search returns chunks
- Ask returns answer + citations + confidence

### Milestone 4 — Delta sync

**Deliverable**

- Manifest-based delta ingest
- Sync button and summary

**Acceptance**

- Modifying one note/file updates only that doc (fast sync)

### Milestone 5 (optional) — One-click Notes export

**Deliverable**

- “Export from Notes” button
- Apple permission flow
- Export → ingest

**Acceptance**

- Fresh install can export and ingest without manual Notes UI export

---

## Implementation Checklist (Engineering Tasks)

### App

- [ ] Choose app shell (Tauri or SwiftUI)
- [ ] Add service manager (spawn backend, port mgmt, health checks)
- [ ] Add setup wizard screens
- [ ] Add Ollama detection + guided install UX
- [ ] Add model pull UI + progress
- [ ] Add folder picker + permissions
- [ ] Add main tabs: Search / Ask / Sync / Health
- [ ] Add logging + error surfaces
- [ ] Add auto-update (optional)

### Backend

- [ ] Add `mode=full|delta` to `/ingest`
- [ ] Add manifest storage and delete handling
- [ ] Ensure `/stats` returns delta summary + last sync
- [ ] Ensure stable citations (chunk_id + source metadata)
- [ ] Package backend as executable (PyInstaller/Nuitka)
- [ ] Document env vars for app control

---

## Risks & Mitigations

- **Ollama installation friction**: provide guided steps and clear detection.
- **Model size/running speed**: default to smaller local models; allow user selection.
- **Apple Notes export limitations**: start with folder ingestion; add direct export later.
- **Context overload for small LLMs**: app/backend must compact context; keep max chunks small.
- **Permissions**: macOS folder + automation permissions need clear messaging.

---

## Open Questions (answer before shipping v1)

1. App shell choice: Tauri or SwiftUI?
2. Default Ollama models: embed model name + chat model name?
3. Supported file formats in export folder (txt/md/html/pdf)?
4. Target macOS version (e.g., 13+ Ventura)?
5. Context window and max token budget tuned for the chosen model?

---
