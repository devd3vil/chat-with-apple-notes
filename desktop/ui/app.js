const DEFAULT_BACKEND_URL = "http://127.0.0.1:8001";
const APP_OWNED_EXPORT_SUBDIR = "NotesLensExport";
const ONBOARDING_EXPORT_LIMIT = null;
const ASK_TIMEOUT_MS = 180000;
const SAVE_THREADS_TIMEOUT_MS = 5000;
const DEFAULT_CONFIG = {
  hasCompletedOnboarding: false,
  exportFolderPath: null,
  lastSyncedAt: null,
  embedModel: "nomic-embed-text",
  chatModel: "neural-chat",
};

const STEP_META = {
  1: { key: "welcome", title: "Welcome" },
  2: { key: "folder", title: "Choose Export Folder" },
  3: { key: "export", title: "Export Progress" },
  4: { key: "setup", title: "Local AI Setup" },
};

const dom = {
  onboardingShell: document.getElementById("onboarding-shell"),
  mainShell: document.getElementById("main-shell"),

  stepLabel: document.getElementById("step-label"),
  stepTitle: document.getElementById("step-title"),
  stepDots: [1, 2, 3, 4].map((idx) => document.getElementById(`step-dot-${idx}`)),
  screens: {
    welcome: document.getElementById("screen-welcome"),
    folder: document.getElementById("screen-folder"),
    export: document.getElementById("screen-export"),
    setup: document.getElementById("screen-setup"),
  },

  startInstallBtn: document.getElementById("start-install"),
  learnMoreBtn: document.getElementById("learn-more"),
  learnMoreCopy: document.getElementById("learn-more-copy"),

  chooseFolderBtn: document.getElementById("choose-folder"),
  folderPath: document.getElementById("folder-path"),
  exportNotesBtn: document.getElementById("export-notes"),
  backWelcomeBtn: document.getElementById("back-welcome"),

  exportStats: document.getElementById("export-stats"),
  exportProgressBar: document.getElementById("export-progress-bar"),
  exportLogLines: document.getElementById("export-log-lines"),
  exportDetails: document.getElementById("export-details"),
  abortExportBtn: document.getElementById("abort-export"),
  nextSetupBtn: document.getElementById("next-setup"),

  embedModelInput: document.getElementById("embed-model"),
  chatModelInput: document.getElementById("chat-model"),
  backExportBtn: document.getElementById("back-export"),
  setupNowBtn: document.getElementById("setup-now"),
  completeSetupBtn: document.getElementById("complete-setup"),
  setupProgressWrap: document.getElementById("setup-progress-wrap"),
  setupStage: document.getElementById("setup-stage"),
  setupProgressBar: document.getElementById("setup-progress-bar"),
  setupMessage: document.getElementById("setup-message"),

  onboardingError: document.getElementById("onboarding-error"),

  syncNotesBtn: document.getElementById("sync-notes"),
  lastSynced: document.getElementById("last-synced"),
  syncDot: document.getElementById("sync-dot"),
  syncBanner: document.getElementById("sync-banner"),
  syncBannerMessage: document.getElementById("sync-banner-message"),
  syncBannerBar: document.getElementById("sync-banner-bar"),
  syncCancelBtn: document.getElementById("sync-cancel"),

  newChatBtn: document.getElementById("new-chat"),
  threadList: document.getElementById("thread-list"),
  transcript: document.getElementById("transcript"),
  composer: document.getElementById("composer"),
  composerInput: document.getElementById("composer-input"),
  composerSend: document.getElementById("composer-send"),
};

const state = {
  baseUrl: DEFAULT_BACKEND_URL,
  config: { ...DEFAULT_CONFIG },
  onboarding: {
    step: 1,
    learnExpanded: false,
    error: "",
    export: {
      status: "idle",
      current: 0,
      total: null,
      message: "",
      logs: [],
      jobId: null,
    },
    setup: {
      busy: false,
      readyToComplete: false,
      progress: 0,
      stage: "Installing Ollama...",
      message: "Preparing local setup...",
      ingestJobId: null,
    },
  },
  threads: [],
  activeThreadId: null,
  sync: {
    busy: false,
    progress: 0,
    message: "",
    exportJobId: null,
    ingestJobId: null,
    cancelRequested: false,
  },
};

class WorkflowCancelled extends Error {}

function tauriInvoke() {
  const tauriGlobal = window.__TAURI__;
  if (typeof tauriGlobal?.invoke === "function") {
    return tauriGlobal.invoke;
  }
  if (typeof tauriGlobal?.tauri?.invoke === "function") {
    return tauriGlobal.tauri.invoke;
  }
  if (typeof tauriGlobal?.core?.invoke === "function") {
    return tauriGlobal.core.invoke;
  }
  return null;
}

function setupFormValue(value) {
  return typeof value === "string" ? value.trim() : "";
}

function nowEpochMsString() {
  return String(Date.now());
}

function toISO(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) {
    return new Date().toISOString();
  }
  return date.toISOString();
}

function normalizeConfig(payload) {
  return {
    hasCompletedOnboarding: Boolean(
      payload?.has_completed_onboarding ??
        payload?.hasCompletedOnboarding ??
        payload?.wizard_completed ??
        payload?.wizardCompleted,
    ),
    exportFolderPath:
      setupFormValue(
        payload?.export_folder_path ??
          payload?.exportFolderPath ??
          payload?.notes_export_dir ??
          payload?.notesExportDir,
      ) || null,
    lastSyncedAt:
      setupFormValue(
        payload?.last_synced_at ??
          payload?.lastSyncedAt ??
          payload?.last_sync_timestamp ??
          payload?.lastSyncTimestamp,
      ) || null,
    embedModel:
      setupFormValue(payload?.embed_model ?? payload?.embedModel) || DEFAULT_CONFIG.embedModel,
    chatModel:
      setupFormValue(payload?.chat_model ?? payload?.chatModel) || DEFAULT_CONFIG.chatModel,
  };
}

function requiresOnboarding(config) {
  return (
    !config.hasCompletedOnboarding ||
    !setupFormValue(config.exportFolderPath) ||
    !setupFormValue(config.lastSyncedAt)
  );
}

function appOwnedExportPath(basePath) {
  return `${setupFormValue(basePath).replace(/\/+$/, "")}/${APP_OWNED_EXPORT_SUBDIR}`;
}

function formatRelativeTime(raw) {
  if (!raw) {
    return "never";
  }
  const numeric = Number(raw);
  const ms = Number.isFinite(numeric) && numeric > 0 ? numeric : Date.parse(raw);
  if (!Number.isFinite(ms)) {
    return "never";
  }
  const sec = Math.max(1, Math.floor((Date.now() - ms) / 1000));
  if (sec < 60) return `${sec} second${sec === 1 ? "" : "s"} ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minute${min === 1 ? "" : "s"} ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} hour${hr === 1 ? "" : "s"} ago`;
  const day = Math.floor(hr / 24);
  return `${day} day${day === 1 ? "" : "s"} ago`;
}

function formatError(err, fallback = "Something went wrong.") {
  if (err instanceof WorkflowCancelled) {
    return "Action cancelled.";
  }
  const raw =
    typeof err === "string"
      ? err
      : typeof err?.message === "string"
        ? err.message
        : String(err ?? "");
  const cleaned = raw.replace(/^Error:\s*/i, "").trim();
  return cleaned || fallback;
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function parseLogs(rawLogs, max = 10) {
  return setupFormValue(rawLogs)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-max);
}

async function parseErrorResponse(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      return body.detail;
    }
    return JSON.stringify(body);
  } catch {
    try {
      return await response.text();
    } catch {
      return `HTTP ${response.status}`;
    }
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await parseErrorResponse(response)}`);
  }
  return response.json();
}

async function postJsonWithTimeout(url, payload, timeoutMs = 90000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await parseErrorResponse(response)}`);
    }
    return response.json();
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error("Request timed out while waiting for the model.");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await parseErrorResponse(response)}`);
  }
  return response.json();
}

function setOnboardingError(message = "") {
  state.onboarding.error = setupFormValue(message);
  renderOnboarding();
}

function goToStep(step) {
  state.onboarding.step = Math.max(1, Math.min(4, step));
  renderOnboarding();
}

function showOnboardingShell() {
  dom.onboardingShell.hidden = false;
  dom.mainShell.hidden = true;
}

function showMainShell() {
  dom.onboardingShell.hidden = true;
  dom.mainShell.hidden = false;
}

function renderStepHeader() {
  const stepMeta = STEP_META[state.onboarding.step];
  dom.stepLabel.textContent = `Step ${state.onboarding.step} of 4`;
  dom.stepTitle.textContent = stepMeta.title;

  dom.stepDots.forEach((dot, idx) => {
    dot.classList.remove("active", "complete");
    const stepNumber = idx + 1;
    if (stepNumber < state.onboarding.step) {
      dot.classList.add("complete");
    } else if (stepNumber === state.onboarding.step) {
      dot.classList.add("active");
    }
  });
}

function renderOnboardingScreens() {
  Object.values(dom.screens).forEach((screen) => screen.classList.remove("active"));
  const stepKey = STEP_META[state.onboarding.step].key;
  dom.screens[stepKey].classList.add("active");

  dom.learnMoreCopy.hidden = !state.onboarding.learnExpanded;

  dom.folderPath.textContent = setupFormValue(state.config.exportFolderPath) || "Not selected";

  const exportState = state.onboarding.export;
  const exportCompleted = exportState.status === "completed";
  const hasTotal = typeof exportState.total === "number" && exportState.total > 0;
  const ratio = hasTotal ? Math.max(0, Math.min(1, exportState.current / exportState.total)) : null;

  if (exportCompleted) {
    dom.exportProgressBar.classList.remove("indeterminate");
    dom.exportProgressBar.style.width = "100%";
    if (hasTotal) {
      dom.exportStats.textContent = `Export completed. ${exportState.total} files exported`;
    } else {
      dom.exportStats.textContent = `Export completed. ${exportState.current} files exported`;
    }
  } else if (hasTotal) {
    dom.exportProgressBar.classList.remove("indeterminate");
    dom.exportProgressBar.style.width = `${Math.max(3, Math.round(ratio * 100))}%`;
    dom.exportStats.textContent = `Exported ${exportState.current} of ${exportState.total} files`;
  } else {
    dom.exportProgressBar.classList.add("indeterminate");
    dom.exportProgressBar.style.width = "28%";
    dom.exportStats.textContent = `Exported ${exportState.current} files`;
  }

  dom.exportLogLines.textContent =
    exportState.logs.length > 0
      ? exportState.logs.join("\n")
      : setupFormValue(exportState.message) || "Logs will appear as notes are exported.";
  if (dom.exportDetails) {
    dom.exportDetails.hidden = exportCompleted;
    if (exportCompleted) {
      dom.exportDetails.open = false;
    }
  }

  const exportRunning = ["queued", "running", "cancelling"].includes(exportState.status);
  dom.abortExportBtn.disabled = !exportRunning;
  dom.nextSetupBtn.disabled = exportState.status !== "completed";

  const setupState = state.onboarding.setup;
  dom.setupProgressWrap.hidden = !setupState.busy && !setupState.readyToComplete;
  dom.setupStage.textContent = setupState.stage;
  dom.setupMessage.textContent = setupState.message;
  dom.setupProgressBar.style.width = `${Math.max(0, Math.min(100, setupState.progress))}%`;
  dom.setupProgressBar.classList.toggle("indeterminate", setupState.busy && setupState.progress < 8);

  dom.embedModelInput.value = state.config.embedModel;
  dom.chatModelInput.value = state.config.chatModel;

  dom.startInstallBtn.disabled = setupState.busy;
  dom.chooseFolderBtn.disabled = exportRunning || setupState.busy;
  dom.exportNotesBtn.disabled = !setupFormValue(state.config.exportFolderPath) || exportRunning || setupState.busy;
  dom.backWelcomeBtn.disabled = exportRunning || setupState.busy;
  dom.backExportBtn.disabled = setupState.busy;
  dom.setupNowBtn.hidden = setupState.readyToComplete;
  dom.setupNowBtn.disabled = setupState.busy || setupState.readyToComplete;
  dom.setupNowBtn.textContent = setupState.busy ? "Setting up..." : "Setup";
  dom.completeSetupBtn.hidden = !setupState.readyToComplete;
  dom.completeSetupBtn.disabled = setupState.busy || !setupState.readyToComplete;

  dom.onboardingError.hidden = !state.onboarding.error;
  dom.onboardingError.textContent = state.onboarding.error;
}

function renderOnboarding() {
  if (dom.onboardingShell.hidden) {
    return;
  }
  renderStepHeader();
  renderOnboardingScreens();
}

function renderSyncUi() {
  const busy = state.sync.busy;
  dom.syncNotesBtn.disabled = busy;
  dom.syncDot.hidden = !busy;
  dom.syncBanner.hidden = !busy;
  if (busy) {
    dom.syncBannerMessage.textContent = state.sync.message || "Sync in progress...";
    dom.syncBannerBar.style.width = `${Math.max(4, Math.min(100, state.sync.progress))}%`;
    dom.syncCancelBtn.hidden = false;
    dom.lastSynced.textContent = `Last synced: ${formatRelativeTime(state.config.lastSyncedAt)}`;
  } else {
    dom.syncCancelBtn.hidden = true;
    dom.syncBannerBar.style.width = "0%";
    dom.lastSynced.textContent = `Last synced: ${formatRelativeTime(state.config.lastSyncedAt)}`;
  }
}

function normalizeThread(raw) {
  return {
    id: raw?.id || `thread-${nowEpochMsString()}`,
    title: setupFormValue(raw?.title) || "New Chat",
    createdAt: setupFormValue(raw?.created_at ?? raw?.createdAt) || nowEpochMsString(),
    updatedAt: setupFormValue(raw?.updated_at ?? raw?.updatedAt) || nowEpochMsString(),
    messages: Array.isArray(raw?.messages)
      ? raw.messages.map((message) => ({
          id: message?.id || `msg-${nowEpochMsString()}-${Math.floor(Math.random() * 10000)}`,
          role: message?.role === "assistant" ? "assistant" : "user",
          text: typeof message?.text === "string" ? message.text : "",
          timestamp: setupFormValue(message?.timestamp) || nowEpochMsString(),
          citations: Array.isArray(message?.citations) ? message.citations : null,
          confidence: typeof message?.confidence === "number" ? message.confidence : null,
          pending: Boolean(message?.pending),
          pendingStage: setupFormValue(message?.pendingStage) || "",
        }))
      : [],
  };
}

function toStorageThread(thread) {
  return {
    id: thread.id,
    title: thread.title,
    created_at: thread.createdAt,
    updated_at: thread.updatedAt,
    messages: thread.messages.map((message) => ({
      id: message.id,
      role: message.role,
      text: message.text,
      timestamp: message.timestamp,
      citations: message.citations ?? null,
      confidence: typeof message.confidence === "number" ? message.confidence : null,
    })),
  };
}

function sortThreads(threads) {
  return [...threads].sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0));
}

function pruneThreads(threads) {
  return sortThreads(threads).slice(0, 10);
}

function currentThread() {
  return state.threads.find((thread) => thread.id === state.activeThreadId) || null;
}

function createThread() {
  const timestamp = nowEpochMsString();
  return {
    id: `thread-${timestamp}-${Math.floor(Math.random() * 10000)}`,
    title: "New Chat",
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
  };
}

function deriveTitle(thread) {
  const firstUser = thread.messages.find((message) => message.role === "user");
  if (!firstUser || !firstUser.text.trim()) {
    return "New Chat";
  }
  return firstUser.text.trim().slice(0, 60);
}

function renderThreads() {
  dom.threadList.innerHTML = "";
  sortThreads(state.threads).forEach((thread) => {
    const item = document.createElement("li");
    item.className = `thread-item${thread.id === state.activeThreadId ? " active" : ""}`;

    const title = document.createElement("h4");
    title.textContent = thread.title;

    const time = document.createElement("span");
    time.textContent = formatRelativeTime(thread.updatedAt);

    item.appendChild(title);
    item.appendChild(time);
    item.addEventListener("click", () => {
      state.activeThreadId = thread.id;
      renderThreads();
      renderTranscript();
    });
    dom.threadList.appendChild(item);
  });
}

function renderTranscript() {
  dom.transcript.innerHTML = "";
  const thread = currentThread();
  if (!thread || thread.messages.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Ask something like: ‘What were my goals for 2026?’";
    dom.transcript.appendChild(empty);
    return;
  }

  thread.messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}${message.pending ? " pending" : ""}`;

    const body = document.createElement("div");
    body.textContent = message.text;
    article.appendChild(body);

    if (message.pending) {
      const pending = document.createElement("div");
      pending.className = "pending-indicator";
      const phase = setupFormValue(message.pendingStage) || "Thinking";
      pending.textContent = `${phase}`;
      const dots = document.createElement("span");
      dots.className = "pending-dots";
      dots.textContent = "...";
      pending.appendChild(dots);
      article.appendChild(pending);
    }

    if (message.role === "assistant" && Array.isArray(message.citations) && message.citations.length > 0) {
      const visibleCitations = message.citations.slice(0, 5);
      const disclosure = document.createElement("details");
      disclosure.className = "citation-disclosure";

      const summary = document.createElement("summary");
      const countLabel = visibleCitations.length === 1 ? "citation" : "citations";
      summary.textContent = `Citations (${visibleCitations.length} ${countLabel})`;
      disclosure.appendChild(summary);

      const citations = document.createElement("ul");
      citations.className = "citation-list";
      visibleCitations.forEach((citation, idx) => {
        const item = document.createElement("li");

        const label = document.createElement("div");
        label.className = "citation-label";

        const noteId = setupFormValue(citation?.note_id ?? citation?.source?.note_id);
        const chunkId = setupFormValue(citation?.chunk_id);
        if (noteId && chunkId) {
          label.textContent = `[${idx + 1}] ${noteId} · ${chunkId}`;
        } else if (noteId) {
          label.textContent = `[${idx + 1}] ${noteId}`;
        } else if (chunkId) {
          label.textContent = `[${idx + 1}] ${chunkId}`;
        } else {
          label.textContent = `[${idx + 1}] Source`;
        }
        item.appendChild(label);

        const snippetText = setupFormValue(citation?.text);
        if (snippetText) {
          const snippet = document.createElement("p");
          snippet.className = "citation-snippet";
          snippet.textContent = snippetText;
          item.appendChild(snippet);
        }

        citations.appendChild(item);
      });
      disclosure.appendChild(citations);
      article.appendChild(disclosure);
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    const confidenceText =
      typeof message.confidence === "number" ? ` · confidence ${Math.round(message.confidence * 100)}%` : "";
    meta.textContent = `${message.role} · ${formatRelativeTime(message.timestamp)}${confidenceText}`;
    article.appendChild(meta);

    dom.transcript.appendChild(article);
  });

  dom.transcript.scrollTop = dom.transcript.scrollHeight;
}

async function resolveBackendUrl() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return DEFAULT_BACKEND_URL;
  }
  try {
    const result = await invoke("backend_base_url");
    if (typeof result === "string" && result.trim()) {
      return result.trim();
    }
  } catch {
    return DEFAULT_BACKEND_URL;
  }
  return DEFAULT_BACKEND_URL;
}

async function loadConfig() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    state.config = { ...DEFAULT_CONFIG };
    return;
  }
  try {
    const loaded = await invoke("load_app_config");
    state.config = normalizeConfig(loaded);
  } catch {
    state.config = { ...DEFAULT_CONFIG };
  }
}

async function saveConfig(updates) {
  state.config = { ...state.config, ...updates };
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return;
  }

  try {
    const saved = await invoke("save_app_config", {
      hasCompletedOnboarding: state.config.hasCompletedOnboarding,
      exportFolderPath: state.config.exportFolderPath,
      lastSyncedAt: state.config.lastSyncedAt,
      embedModel: state.config.embedModel,
      chatModel: state.config.chatModel,
    });
    state.config = normalizeConfig(saved);
  } catch {
    // Keep local state for non-desktop shell fallback.
  }
}

async function loadThreads() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    state.threads = [createThread()];
    state.activeThreadId = state.threads[0].id;
    return;
  }

  try {
    const loaded = await invoke("load_chat_threads");
    state.threads = Array.isArray(loaded) ? loaded.map(normalizeThread) : [];
  } catch {
    state.threads = [];
  }

  if (state.threads.length === 0) {
    state.threads = [createThread()];
  }
  state.threads = pruneThreads(state.threads);
  state.activeThreadId = state.threads[0].id;
}

async function saveThreads() {
  state.threads = pruneThreads(state.threads);
  if (!state.threads.some((thread) => thread.id === state.activeThreadId)) {
    state.activeThreadId = state.threads[0]?.id || null;
  }

  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return;
  }

  let timer = null;
  const timeoutPromise = new Promise((_, reject) => {
    timer = window.setTimeout(
      () => reject(new Error("save_chat_threads timed out")),
      SAVE_THREADS_TIMEOUT_MS,
    );
  });

  try {
    await Promise.race([
      invoke("save_chat_threads", {
        threads: state.threads.map(toStorageThread),
      }),
      timeoutPromise,
    ]);
  } catch {
    // Keep in-memory state even if persistence fails.
  } finally {
    if (timer) {
      window.clearTimeout(timer);
    }
  }
}

async function chooseFolder() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setOnboardingError("Folder picker requires desktop shell.");
    return;
  }

  try {
    const selected = await invoke("pick_export_folder");
    const folder = setupFormValue(typeof selected === "string" ? selected : selected?.path);
    if (!folder) {
      return;
    }
    await saveConfig({ exportFolderPath: folder });
    state.onboarding.export = {
      status: "idle",
      current: 0,
      total: null,
      message: "",
      logs: [],
      jobId: null,
    };
    setOnboardingError("");
  } catch (err) {
    setOnboardingError(formatError(err, "Could not open folder picker."));
  }
}

async function pollExportJob() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    throw new Error("Export requires desktop shell.");
  }

  while (true) {
    const snapshot = await invoke("get_export_job");
    if (snapshot && typeof snapshot === "object") {
      state.onboarding.export.status = setupFormValue(snapshot.status) || "running";
      state.onboarding.export.current = Number(snapshot.current || 0);
      state.onboarding.export.total =
        typeof snapshot.total === "number" && snapshot.total >= 0 ? snapshot.total : null;
      state.onboarding.export.message = setupFormValue(snapshot.message) || "Exporting...";
      state.onboarding.export.logs = parseLogs(snapshot.logs, 10);

      if (
        state.onboarding.export.status !== "completed" &&
        state.onboarding.export.message.toLowerCase().includes("export complete")
      ) {
        state.onboarding.export.status = "completed";
      }
      renderOnboarding();

      if (["completed", "failed", "cancelled"].includes(state.onboarding.export.status)) {
        return snapshot;
      }
    }
    await delay(500);
  }
}

async function startExportFlow() {
  const folder = setupFormValue(state.config.exportFolderPath);
  if (!folder) {
    setOnboardingError("Choose a folder before exporting notes.");
    return;
  }

  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setOnboardingError("Export requires desktop shell.");
    return;
  }

  setOnboardingError("");
  goToStep(3);
  state.onboarding.export = {
    status: "queued",
    current: 0,
    total: null,
    message: "Starting export...",
    logs: ["Starting export..."],
    jobId: null,
  };
  renderOnboarding();

  try {
    // Ensure each onboarding export run starts with a clean app-owned subfolder.
    await cleanupExportAfterAbort();
    const exportPayload = {
      exportFolderPath: folder,
    };
    if (Number.isInteger(ONBOARDING_EXPORT_LIMIT) && ONBOARDING_EXPORT_LIMIT > 0) {
      exportPayload.maxFiles = ONBOARDING_EXPORT_LIMIT;
    }
    const started = await invoke("start_export_job", exportPayload);
    state.onboarding.export.jobId = started?.job_id || null;

    const terminal = await pollExportJob();
    const status =
      state.onboarding.export.status === "completed"
        ? "completed"
        : setupFormValue(terminal?.status || state.onboarding.export.status);

    if (status === "completed") {
      state.onboarding.export.status = "completed";
      state.onboarding.export.message = "Export finished successfully.";
      renderOnboarding();
      await delay(300);
      goToStep(4);
      return;
    }

    if (status === "cancelled") {
      throw new WorkflowCancelled("Export cancelled.");
    }

    throw new Error(setupFormValue(terminal?.error) || setupFormValue(terminal?.message) || "Export failed.");
  } catch (err) {
    await cleanupExportAfterAbort();
    state.onboarding.export = {
      status: "idle",
      current: 0,
      total: null,
      message: "",
      logs: [],
      jobId: null,
    };
    goToStep(2);
    setOnboardingError(formatError(err, "Export failed. Please try again."));
  }
}

async function cleanupExportAfterAbort() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return;
  }
  const folder = setupFormValue(state.config.exportFolderPath);
  if (!folder) {
    return;
  }

  await invoke("delete_export_subfolder", {
    exportFolderPath: folder,
  }).catch(() => null);
}

async function abortExportFlow() {
  const running = ["queued", "running", "cancelling"].includes(state.onboarding.export.status);
  if (!running) {
    return;
  }
  const shouldAbort = window.confirm("Abort export and clean up partial files?");
  if (!shouldAbort) {
    return;
  }

  const invoke = tauriInvoke();
  if (typeof invoke === "function") {
    await invoke("cancel_export_job").catch(() => null);
  }
  await cleanupExportAfterAbort();

  state.onboarding.export = {
    status: "idle",
    current: 0,
    total: null,
    message: "",
    logs: [],
    jobId: null,
  };
  goToStep(2);
  setOnboardingError("Export aborted. You can choose a folder and try again.");
}

async function ensureBackendHealthy() {
  try {
    await getJson(`${state.baseUrl}/health`);
  } catch {
    throw new Error("Backend unavailable. Start the app backend and retry.");
  }
}

async function ensureChatReady() {
  let health;
  try {
    health = await getJson(`${state.baseUrl}/health`);
  } catch {
    throw new Error("Backend unavailable. Start the app backend and retry.");
  }

  if (health?.status !== "ok") {
    throw new Error("Backend is not ready yet.");
  }
  if (health?.llm === false) {
    throw new Error(
      `Local chat model is unavailable. Install '${state.config.chatModel}' in Ollama and retry.`,
    );
  }
  if (health?.embedder === false) {
    throw new Error(
      `Embedding model is unavailable. Install '${state.config.embedModel}' in Ollama and retry.`,
    );
  }
}

async function runSetupStage(label, fromProgress, toProgress, task, minMs = 1200) {
  state.onboarding.setup.stage = label;
  state.onboarding.setup.message = label;
  state.onboarding.setup.progress = Math.max(state.onboarding.setup.progress, fromProgress);
  renderOnboarding();

  const rampCap = Math.max(fromProgress, toProgress - 4);
  const timer = window.setInterval(() => {
    if (!state.onboarding.setup.busy) {
      return;
    }
    if (state.onboarding.setup.progress < rampCap) {
      state.onboarding.setup.progress = Math.min(rampCap, state.onboarding.setup.progress + 1);
      renderOnboarding();
    }
  }, 140);

  const startedAt = Date.now();
  let stageError = null;
  try {
    await task();
  } catch (err) {
    stageError = err;
  } finally {
    window.clearInterval(timer);
  }

  const elapsed = Date.now() - startedAt;
  if (elapsed < minMs) {
    await delay(minMs - elapsed);
  }
  if (!stageError) {
    state.onboarding.setup.progress = Math.max(state.onboarding.setup.progress, toProgress);
  }
  renderOnboarding();

  if (stageError) {
    throw stageError;
  }
}

async function ensureModelAvailable(model, installedModels) {
  if (installedModels.includes(model)) {
    return;
  }

  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    throw new Error("Model installation requires desktop shell.");
  }

  const pull = await invoke("ollama_pull_model", { model });
  if (!pull?.success) {
    throw new Error(`Model pull failed for ${model}. ${setupFormValue(pull?.output)}`.trim());
  }
}

async function runIngestSetupPhase(folderPath) {
  const payload = await postJson(`${state.baseUrl}/jobs/ingest`, {
    export_dir: appOwnedExportPath(folderPath),
    mode: "full",
    reindex: true,
  });
  state.onboarding.setup.ingestJobId = payload.job_id;

  while (true) {
    const snapshot = await getJson(`${state.baseUrl}/jobs/${payload.job_id}`);
    state.onboarding.setup.message = setupFormValue(snapshot.message) || "Warming up...";

    if (typeof snapshot.total === "number" && snapshot.total > 0) {
      const ratio = Math.max(0, Math.min(1, Number(snapshot.current || 0) / snapshot.total));
      const mapped = Math.round(70 + ratio * 30);
      state.onboarding.setup.progress = Math.max(state.onboarding.setup.progress, Math.min(99, mapped));
    }
    renderOnboarding();

    if (snapshot.status === "completed") {
      state.onboarding.setup.progress = 100;
      renderOnboarding();
      return snapshot;
    }
    if (snapshot.status === "cancelled") {
      throw new WorkflowCancelled("Setup was cancelled.");
    }
    if (snapshot.status === "failed") {
      throw new Error(setupFormValue(snapshot.error) || "Indexing failed.");
    }

    await delay(600);
  }
}

async function startLocalSetupFlow() {
  if (state.onboarding.setup.busy) {
    return;
  }

  const folderPath = setupFormValue(state.config.exportFolderPath);
  if (!folderPath) {
    setOnboardingError("Choose an export folder before setup.");
    goToStep(2);
    return;
  }
  if (state.onboarding.export.status !== "completed") {
    setOnboardingError("Complete note export first.");
    goToStep(3);
    return;
  }

  const embedModel = setupFormValue(dom.embedModelInput.value) || DEFAULT_CONFIG.embedModel;
  const chatModel = setupFormValue(dom.chatModelInput.value) || DEFAULT_CONFIG.chatModel;

  await saveConfig({ embedModel, chatModel, exportFolderPath: folderPath });

  state.onboarding.setup.busy = true;
  state.onboarding.setup.readyToComplete = false;
  state.onboarding.setup.progress = 2;
  state.onboarding.setup.stage = "Installing Ollama...";
  state.onboarding.setup.message = "Preparing local setup...";
  setOnboardingError("");
  renderOnboarding();

  try {
    await runSetupStage(
      "Installing Ollama...",
      2,
      18,
      async () => {
        await ensureBackendHealthy();
        const invoke = tauriInvoke();
        if (typeof invoke !== "function") {
          throw new Error("Ollama checks require desktop shell.");
        }
        let ollama = await invoke("ollama_status");
        if (!ollama?.installed) {
          throw new Error("Ollama is not installed. Install Ollama and click Setup again.");
        }
        if (!ollama?.running) {
          await invoke("start_ollama").catch(() => null);
          await delay(1200);
          ollama = await invoke("ollama_status");
          if (!ollama?.running) {
            throw new Error("Ollama is installed but not running. Open Ollama and retry.");
          }
        }
      },
      1800,
    );

    let installedModels = [];
    const invoke = tauriInvoke();
    if (typeof invoke === "function") {
      const status = await invoke("ollama_status");
      installedModels = Array.isArray(status?.models) ? status.models : [];
    }

    await runSetupStage(
      "Downloading chat model...",
      18,
      45,
      async () => {
        await ensureModelAvailable(chatModel, installedModels);
      },
      2000,
    );

    if (!installedModels.includes(chatModel)) {
      installedModels.push(chatModel);
    }

    await runSetupStage(
      "Downloading embedding model...",
      45,
      70,
      async () => {
        await ensureModelAvailable(embedModel, installedModels);
      },
      2200,
    );

    state.onboarding.setup.stage = "Warming up...";
    state.onboarding.setup.message = "Indexing your notes locally...";
    renderOnboarding();
    const ingestResult = await runIngestSetupPhase(folderPath);

    await saveConfig({
      hasCompletedOnboarding: true,
      exportFolderPath: folderPath,
      lastSyncedAt: toISO(ingestResult?.result?.last_sync_time),
    });

    state.onboarding.setup.busy = false;
    state.onboarding.setup.readyToComplete = true;
    state.onboarding.setup.progress = 100;
    state.onboarding.setup.stage = "Setup complete";
    state.onboarding.setup.message = "Local setup is ready. Click Complete to open chat.";
    renderOnboarding();
  } catch (err) {
    state.onboarding.setup.busy = false;
    state.onboarding.setup.readyToComplete = false;
    renderOnboarding();
    setOnboardingError(formatError(err, "Setup failed. Please retry."));
  }
}

async function completeSetupFlow() {
  if (state.onboarding.setup.busy || !state.onboarding.setup.readyToComplete) {
    return;
  }

  await loadThreads();
  renderThreads();
  renderTranscript();
  renderSyncUi();
  showMainShell();
}

async function startSyncFlow() {
  if (state.sync.busy) {
    return;
  }

  const folderPath = setupFormValue(state.config.exportFolderPath);
  if (!folderPath) {
    state.sync.message = "No export folder configured.";
    renderSyncUi();
    return;
  }

  state.sync.busy = true;
  state.sync.cancelRequested = false;
  state.sync.progress = 4;
  state.sync.message = "Exporting latest notes...";
  renderSyncUi();

  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    state.sync.busy = false;
    state.sync.message = "Sync requires desktop shell.";
    renderSyncUi();
    return;
  }

  try {
    // Keep sync export deterministic by removing stale files before re-export.
    await invoke("delete_export_subfolder", {
      exportFolderPath: folderPath,
    }).catch(() => null);

    const exportStart = await invoke("start_export_job", {
      exportFolderPath: folderPath,
    });
    state.sync.exportJobId = exportStart?.job_id || null;

    while (true) {
      if (state.sync.cancelRequested) {
        throw new WorkflowCancelled("Sync cancelled.");
      }
      const snapshot = await invoke("get_export_job");
      if (snapshot && typeof snapshot === "object") {
        const total = typeof snapshot.total === "number" && snapshot.total > 0 ? snapshot.total : null;
        if (total) {
          const ratio = Math.max(0, Math.min(1, Number(snapshot.current || 0) / total));
          state.sync.progress = Math.max(state.sync.progress, Math.round(ratio * 50));
        } else {
          state.sync.progress = Math.min(45, state.sync.progress + 1);
        }
        state.sync.message = setupFormValue(snapshot.message) || "Exporting latest notes...";
        renderSyncUi();

        if (["completed", "failed", "cancelled"].includes(snapshot.status)) {
          if (snapshot.status === "completed") {
            break;
          }
          if (snapshot.status === "cancelled") {
            throw new WorkflowCancelled("Sync cancelled.");
          }
          throw new Error(setupFormValue(snapshot.error) || "Export sync failed.");
        }
      }
      await delay(500);
    }

    state.sync.message = "Indexing note changes...";
    state.sync.progress = Math.max(state.sync.progress, 52);
    renderSyncUi();

    const ingestStart = await postJson(`${state.baseUrl}/jobs/ingest`, {
      export_dir: appOwnedExportPath(folderPath),
      mode: "delta",
      reindex: false,
    });
    state.sync.ingestJobId = ingestStart.job_id;

    while (true) {
      if (state.sync.cancelRequested) {
        throw new WorkflowCancelled("Sync cancelled.");
      }

      const snapshot = await getJson(`${state.baseUrl}/jobs/${ingestStart.job_id}`);
      state.sync.message = setupFormValue(snapshot.message) || "Indexing note changes...";

      if (typeof snapshot.total === "number" && snapshot.total > 0) {
        const ratio = Math.max(0, Math.min(1, Number(snapshot.current || 0) / snapshot.total));
        state.sync.progress = Math.max(state.sync.progress, 50 + Math.round(ratio * 50));
      } else {
        state.sync.progress = Math.min(96, state.sync.progress + 1);
      }
      renderSyncUi();

      if (snapshot.status === "completed") {
        state.sync.progress = 100;
        renderSyncUi();
        await saveConfig({ lastSyncedAt: toISO(snapshot?.result?.last_sync_time) });
        break;
      }
      if (snapshot.status === "cancelled") {
        throw new WorkflowCancelled("Sync cancelled.");
      }
      if (snapshot.status === "failed") {
        throw new Error(setupFormValue(snapshot.error) || "Sync ingest failed.");
      }

      await delay(600);
    }

    state.sync.message = "Sync complete.";
    renderSyncUi();
    await delay(500);
  } catch (err) {
    state.sync.message = formatError(err, "Sync failed.");
    renderSyncUi();
  } finally {
    state.sync.busy = false;
    state.sync.progress = 0;
    state.sync.exportJobId = null;
    state.sync.ingestJobId = null;
    state.sync.cancelRequested = false;
    renderSyncUi();
  }
}

async function cancelSyncFlow() {
  if (!state.sync.busy) {
    return;
  }
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return;
  }

  state.sync.cancelRequested = true;
  await invoke("cancel_export_job").catch(() => null);
  if (state.sync.ingestJobId) {
    await postJson(`${state.baseUrl}/jobs/${state.sync.ingestJobId}/cancel`, {}).catch(() => null);
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const query = setupFormValue(dom.composerInput.value);
  if (!query) {
    return;
  }

  let thread = currentThread();
  if (!thread) {
    thread = createThread();
    state.threads.unshift(thread);
    state.activeThreadId = thread.id;
  }

  thread.messages.push({
    id: `msg-${nowEpochMsString()}-u`,
    role: "user",
    text: query,
    timestamp: nowEpochMsString(),
  });

  const assistantMessage = {
    id: `msg-${nowEpochMsString()}-a-pending`,
    role: "assistant",
    text: "Retrieving relevant notes...",
    timestamp: nowEpochMsString(),
    citations: null,
    confidence: null,
    pending: true,
    pendingStage: "Retrieving",
  };
  thread.messages.push(assistantMessage);
  thread.updatedAt = nowEpochMsString();
  thread.title = deriveTitle(thread);

  dom.composerInput.value = "";
  dom.composerSend.disabled = true;
  renderThreads();
  renderTranscript();
  void saveThreads();

  let phaseTimer = null;
  let slowTimer = null;
  try {
    phaseTimer = window.setTimeout(() => {
      assistantMessage.pendingStage = "Thinking";
      assistantMessage.text = "Thinking through your notes...";
      renderTranscript();
    }, 900);

    slowTimer = window.setTimeout(() => {
      assistantMessage.pendingStage = "Still working";
      assistantMessage.text = "Running your local model. This can take longer on first response...";
      renderTranscript();
    }, 20000);

    await ensureChatReady();
    const response = await postJsonWithTimeout(`${state.baseUrl}/ask`, { query }, ASK_TIMEOUT_MS);
    assistantMessage.id = `msg-${nowEpochMsString()}-a`;
    assistantMessage.pending = false;
    assistantMessage.pendingStage = "";
    assistantMessage.text =
      setupFormValue(response.answer) || "I could not generate an answer from your notes.";
    assistantMessage.timestamp = nowEpochMsString();
    assistantMessage.citations = Array.isArray(response.citations) ? response.citations : null;
    assistantMessage.confidence =
      typeof response.confidence === "number" ? response.confidence : null;
  } catch (err) {
    assistantMessage.id = `msg-${nowEpochMsString()}-a`;
    assistantMessage.pending = false;
    assistantMessage.pendingStage = "";
    assistantMessage.text = `Request failed: ${formatError(err)}`;
    assistantMessage.timestamp = nowEpochMsString();
    assistantMessage.citations = null;
    assistantMessage.confidence = null;
  } finally {
    if (phaseTimer) {
      window.clearTimeout(phaseTimer);
    }
    if (slowTimer) {
      window.clearTimeout(slowTimer);
    }
    thread.updatedAt = nowEpochMsString();
    thread.title = deriveTitle(thread);
    dom.composerSend.disabled = false;
    renderThreads();
    renderTranscript();
    await saveThreads();
  }
}

function wireEvents() {
  dom.startInstallBtn.addEventListener("click", () => {
    setOnboardingError("");
    goToStep(2);
  });

  dom.learnMoreBtn.addEventListener("click", () => {
    state.onboarding.learnExpanded = !state.onboarding.learnExpanded;
    renderOnboarding();
  });

  dom.chooseFolderBtn.addEventListener("click", chooseFolder);
  dom.exportNotesBtn.addEventListener("click", startExportFlow);
  dom.backWelcomeBtn.addEventListener("click", () => {
    setOnboardingError("");
    goToStep(1);
  });

  dom.abortExportBtn.addEventListener("click", abortExportFlow);
  dom.nextSetupBtn.addEventListener("click", () => {
    setOnboardingError("");
    goToStep(4);
  });

  dom.backExportBtn.addEventListener("click", () => {
    setOnboardingError("");
    goToStep(3);
  });
  dom.setupNowBtn.addEventListener("click", startLocalSetupFlow);
  dom.completeSetupBtn.addEventListener("click", completeSetupFlow);

  dom.syncNotesBtn.addEventListener("click", startSyncFlow);
  dom.syncCancelBtn.addEventListener("click", cancelSyncFlow);

  dom.newChatBtn.addEventListener("click", async () => {
    const thread = createThread();
    state.threads.unshift(thread);
    state.threads = pruneThreads(state.threads);
    state.activeThreadId = thread.id;
    renderThreads();
    renderTranscript();
    await saveThreads();
  });

  dom.composer.addEventListener("submit", sendMessage);
  dom.composerInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      dom.composer.requestSubmit();
    }
  });
}

async function bootstrap() {
  wireEvents();

  state.baseUrl = await resolveBackendUrl();
  await loadConfig();

  if (requiresOnboarding(state.config)) {
    state.onboarding.step = 1;
    state.onboarding.learnExpanded = false;
    showOnboardingShell();
    renderOnboarding();
    return;
  }

  await loadThreads();
  renderThreads();
  renderTranscript();
  renderSyncUi();
  showMainShell();
}

bootstrap();
