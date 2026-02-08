const DEFAULT_BACKEND_URL = "http://127.0.0.1:8001";
const DEFAULT_SETUP_CONFIG = {
  embed_model: "nomic-embed-text",
  chat_model: "neural-chat",
  notes_export_dir: null,
  wizard_completed: false,
};

const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const backendUrlText = document.getElementById("backend-url");
const healthOutput = document.getElementById("health-output");
const refreshHealthBtn = document.getElementById("refresh");

const ollamaSummary = document.getElementById("ollama-summary");
const ollamaOutput = document.getElementById("ollama-output");
const wizardOutput = document.getElementById("wizard-output");
const checkOllamaBtn = document.getElementById("check-ollama");
const installOllamaBtn = document.getElementById("install-ollama");
const startOllamaBtn = document.getElementById("start-ollama");
const saveSetupBtn = document.getElementById("save-setup");
const pullModelsBtn = document.getElementById("pull-models");
const retryFailedBtn = document.getElementById("retry-failed");
const resumePullBtn = document.getElementById("resume-pull");
const embedModelInput = document.getElementById("embed-model");
const chatModelInput = document.getElementById("chat-model");

const exportFolderPath = document.getElementById("export-folder-path");
const pickFolderBtn = document.getElementById("pick-folder");
const validateFolderBtn = document.getElementById("validate-folder");
const ingestFullBtn = document.getElementById("ingest-full");
const syncNowBtn = document.getElementById("sync-now");
const ingestOutput = document.getElementById("ingest-output");

const searchQueryInput = document.getElementById("search-query");
const searchTopKInput = document.getElementById("search-top-k");
const runSearchBtn = document.getElementById("run-search");
const searchResults = document.getElementById("search-results");

const askQueryInput = document.getElementById("ask-query");
const runAskBtn = document.getElementById("run-ask");
const askAnswer = document.getElementById("ask-answer");
const askConfidence = document.getElementById("ask-confidence");
const askCitations = document.getElementById("ask-citations");

const refreshStatsBtn = document.getElementById("refresh-stats");
const statsOutput = document.getElementById("stats-output");

let healthUrl = `${DEFAULT_BACKEND_URL}/health`;
let baseUrl = DEFAULT_BACKEND_URL;
let setupState = { ...DEFAULT_SETUP_CONFIG };
let shellAvailable = true;
let healthPollTimer = null;
let ingestInProgress = false;

const pullState = {
  inProgress: false,
  completed: [],
  failed: [],
  pending: [],
};

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

function truncateText(text, maxChars = 3500) {
  if (typeof text !== "string") {
    return String(text);
  }
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, maxChars)}\n\n...truncated...`;
}

function setupFormValue(value) {
  return typeof value === "string" ? value.trim() : "";
}

function readCommandField(payload, snakeKey, camelKey) {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  if (payload[snakeKey] !== undefined) {
    return payload[snakeKey];
  }
  return payload[camelKey];
}

function normalizeSetupConfig(payload) {
  const embedModel = setupFormValue(
    readCommandField(payload, "embed_model", "embedModel") || DEFAULT_SETUP_CONFIG.embed_model,
  );
  const chatModel = setupFormValue(
    readCommandField(payload, "chat_model", "chatModel") || DEFAULT_SETUP_CONFIG.chat_model,
  );
  const notesExportDir = setupFormValue(
    readCommandField(payload, "notes_export_dir", "notesExportDir"),
  );
  const wizardCompleted = readCommandField(payload, "wizard_completed", "wizardCompleted");
  return {
    embed_model: embedModel || DEFAULT_SETUP_CONFIG.embed_model,
    chat_model: chatModel || DEFAULT_SETUP_CONFIG.chat_model,
    notes_export_dir: notesExportDir || null,
    wizard_completed: Boolean(wizardCompleted),
  };
}

function resolveSelectedExportFolder() {
  const fromState = setupFormValue(setupState.notes_export_dir);
  if (fromState) {
    return fromState;
  }
  const fromDisplay = setupFormValue(exportFolderPath?.textContent);
  if (fromDisplay && fromDisplay !== "Not selected") {
    return fromDisplay;
  }
  return "";
}

function uniqueStrings(values) {
  const output = [];
  values.forEach((value) => {
    if (value && !output.includes(value)) {
      output.push(value);
    }
  });
  return output;
}

function createCard(title, subtitle, text) {
  const card = document.createElement("div");
  card.className = "result-item";

  const heading = document.createElement("h4");
  heading.textContent = title;
  card.appendChild(heading);

  if (subtitle) {
    const meta = document.createElement("p");
    meta.className = "backend-url";
    meta.textContent = subtitle;
    card.appendChild(meta);
  }

  const body = document.createElement("pre");
  body.textContent = text;
  card.appendChild(body);

  return card;
}

function setStatusView(mode, text) {
  statusText.textContent = text;
  if (mode === "healthy") {
    statusDot.style.background = "#2f9e44";
  } else if (mode === "warn") {
    statusDot.style.background = "#f59f00";
  } else {
    statusDot.style.background = "#d9480f";
  }
}

function setWizardMessage(message, append = false) {
  if (!wizardOutput) {
    return;
  }
  wizardOutput.textContent = append
    ? `${wizardOutput.textContent}\n${message}`
    : message;
}

function setIngestMessage(message, append = false) {
  if (!ingestOutput) {
    return;
  }
  ingestOutput.textContent = append
    ? `${ingestOutput.textContent}\n${message}`
    : message;
}

function renderExportValidation(status) {
  if (!status) {
    setIngestMessage("Folder validation returned no status.");
    return;
  }
  let output = JSON.stringify(status, null, 2);
  if (status.permission_denied) {
    output +=
      "\n\nPermission guidance:\n1) Open macOS System Settings > Privacy & Security.\n2) Grant this app (or Terminal in dev mode) access to the selected folder.\n3) Click Validate Folder again.";
  }
  setIngestMessage(output);
}

function setSetupForm(config) {
  if (embedModelInput) {
    embedModelInput.value = config.embed_model || DEFAULT_SETUP_CONFIG.embed_model;
  }
  if (chatModelInput) {
    chatModelInput.value = config.chat_model || DEFAULT_SETUP_CONFIG.chat_model;
  }
}

function updateExportFolderDisplay() {
  if (!exportFolderPath) {
    return;
  }
  const path = setupState.notes_export_dir;
  exportFolderPath.textContent = path && path.trim() ? path : "Not selected";
}

function updateWizardButtons() {
  const disable = pullState.inProgress || !shellAvailable;
  [checkOllamaBtn, installOllamaBtn, startOllamaBtn, saveSetupBtn, pullModelsBtn].forEach(
    (button) => {
      if (button) {
        button.disabled = disable;
      }
    },
  );
  if (retryFailedBtn) {
    retryFailedBtn.disabled = disable || pullState.failed.length === 0;
  }
  if (resumePullBtn) {
    resumePullBtn.disabled = disable || pullState.pending.length === 0;
  }
}

function updateFolderActionButtons() {
  const disable = !shellAvailable || ingestInProgress;
  [pickFolderBtn, validateFolderBtn, ingestFullBtn, syncNowBtn].forEach((button) => {
    if (button) {
      button.disabled = disable;
    }
  });
}

function startHealthPolling() {
  if (healthPollTimer !== null) {
    return;
  }
  healthPollTimer = window.setInterval(fetchHealth, 5000);
}

function stopHealthPolling() {
  if (healthPollTimer === null) {
    return;
  }
  window.clearInterval(healthPollTimer);
  healthPollTimer = null;
}

function renderOllamaStatus(status) {
  if (!status) {
    return;
  }
  if (ollamaSummary) {
    if (!status.installed) {
      ollamaSummary.textContent = "Not installed";
    } else if (!status.running) {
      ollamaSummary.textContent = "Installed, not running";
    } else {
      ollamaSummary.textContent = "Running";
    }
  }
  if (ollamaOutput) {
    ollamaOutput.textContent = JSON.stringify(
      {
        installed: status.installed,
        running: status.running,
        version: status.version,
        message: status.message,
        installed_models: status.models,
      },
      null,
      2,
    );
  }
}

function switchTab(target) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tabTarget === target);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${target}`);
  });
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
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return response.json();
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return response.json();
}

async function resolveBackendUrl() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    shellAvailable = false;
    return DEFAULT_BACKEND_URL;
  }
  try {
    const resolved = await invoke("backend_base_url");
    if (typeof resolved === "string" && resolved.trim()) {
      return resolved.trim();
    }
  } catch (err) {
    healthOutput.textContent = `Could not resolve backend URL from app shell: ${err}`;
  }
  return DEFAULT_BACKEND_URL;
}

async function fetchHealth() {
  setStatusView("warn", "Checking...");
  try {
    const data = await getJson(`${baseUrl}/health`);
    healthOutput.textContent = JSON.stringify(data, null, 2);
    const healthy = data.status === "ok" && data.store;
    setStatusView(healthy ? "healthy" : "warn", healthy ? "Healthy" : "Degraded");
  } catch (err) {
    healthOutput.textContent = `Health check failed: ${err}`;
    setStatusView("offline", "Offline");
  }
}

async function refreshStats() {
  try {
    const stats = await getJson(`${baseUrl}/stats`);
    statsOutput.textContent = JSON.stringify(stats, null, 2);
  } catch (err) {
    statsOutput.textContent = `Stats request failed: ${err}`;
  }
}

async function loadSetupConfig() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    shellAvailable = false;
    setSetupForm(DEFAULT_SETUP_CONFIG);
    setWizardMessage("Setup wizard actions require the Tauri desktop shell.");
    setupState = { ...DEFAULT_SETUP_CONFIG };
    updateExportFolderDisplay();
    updateWizardButtons();
    updateFolderActionButtons();
    return setupState;
  }

  try {
    const loaded = await invoke("setup_load_config");
    setupState = normalizeSetupConfig(loaded);
    setSetupForm(setupState);
    updateExportFolderDisplay();
    setWizardMessage(
      setupState.wizard_completed
        ? "Setup config loaded. Wizard is marked complete."
        : "Setup config loaded. Continue with setup steps.",
    );
    return setupState;
  } catch (err) {
    setupState = { ...DEFAULT_SETUP_CONFIG };
    setSetupForm(setupState);
    updateExportFolderDisplay();
    setWizardMessage(`Could not load setup config: ${err}`);
    return setupState;
  } finally {
    updateWizardButtons();
    updateFolderActionButtons();
  }
}

async function saveSetupConfig(wizardCompleted) {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    throw new Error("Tauri invoke is not available");
  }

  const embedModel = setupFormValue(embedModelInput?.value);
  const chatModel = setupFormValue(chatModelInput?.value);
  if (!embedModel || !chatModel) {
    throw new Error("Both embedding and chat model names are required");
  }

  const saved = await invoke("setup_save_config", {
    embedModel,
    chatModel,
    wizardCompleted,
    embed_model: embedModel,
    chat_model: chatModel,
    wizard_completed: wizardCompleted,
  });
  setupState = normalizeSetupConfig(saved);
  updateExportFolderDisplay();
  setWizardMessage(
    `Saved setup config (embed=${setupState.embed_model}, chat=${setupState.chat_model}, completed=${setupState.wizard_completed})`,
  );
  return setupState;
}

async function saveExportFolder(path) {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    throw new Error("Tauri invoke is not available");
  }
  const normalizedPath = setupFormValue(path);
  const saved = await invoke("setup_save_notes_export_dir", {
    notesExportDir: normalizedPath,
    notes_export_dir: normalizedPath,
  });
  setupState = normalizeSetupConfig(saved);
  updateExportFolderDisplay();
  return setupState;
}

async function checkOllamaStatus() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setWizardMessage("Cannot check Ollama: Tauri invoke is unavailable.");
    return;
  }
  try {
    const status = await invoke("ollama_status");
    renderOllamaStatus(status);
    setWizardMessage(status.message);
  } catch (err) {
    setWizardMessage(`Failed to check Ollama: ${err}`);
  }
}

async function installOllama() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setWizardMessage("Cannot open installer: Tauri invoke is unavailable.");
    return;
  }
  try {
    await invoke("open_ollama_download_page");
    setWizardMessage("Opened Ollama download page in your browser.");
  } catch (err) {
    setWizardMessage(`Failed to open Ollama download page: ${err}`);
  }
}

async function startOllama() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setWizardMessage("Cannot start Ollama: Tauri invoke is unavailable.");
    return;
  }
  try {
    const message = await invoke("start_ollama");
    setWizardMessage(message);
    await checkOllamaStatus();
  } catch (err) {
    setWizardMessage(`Failed to start Ollama: ${err}`);
  }
}

function describePullFailure(result) {
  if (!result || result.success) {
    return "";
  }
  switch (result.error_type) {
    case "timeout":
      return "Timed out while pulling model.";
    case "offline":
      return "Network/Ollama connectivity issue detected.";
    case "disk_full":
      return "Insufficient disk space detected.";
    case "not_installed":
      return "Ollama is not installed or unavailable in PATH.";
    default:
      return "Unknown pull error.";
  }
}

async function executePullPlan(models, label) {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setWizardMessage("Cannot pull models: Tauri invoke is unavailable.");
    return false;
  }

  const plan = uniqueStrings(models);
  if (plan.length === 0) {
    setWizardMessage("No models queued for pull.");
    return false;
  }

  pullState.failed = pullState.failed.filter((entry) => !plan.includes(entry.model));
  pullState.pending = [...plan];
  pullState.inProgress = true;
  updateWizardButtons();

  try {
    setWizardMessage(`${label}: ${plan.join(", ")}`, true);
    for (let idx = 0; idx < plan.length; idx += 1) {
      const model = plan[idx];
      setWizardMessage(`Pulling ${model}...`, true);
      const result = await invoke("ollama_pull_model", { model });
      pullState.pending = plan.slice(idx + 1);

      if (result.success) {
        if (!pullState.completed.includes(model)) {
          pullState.completed.push(model);
        }
        setWizardMessage(`Model ${model}: success`, true);
        continue;
      }

      pullState.failed = pullState.failed.filter((entry) => entry.model !== model);
      pullState.failed.push({
        model,
        error_type: result.error_type || "unknown",
        retryable: Boolean(result.retryable),
      });

      setWizardMessage(
        `Model ${model}: failed (${result.error_type || "unknown"}, retryable=${Boolean(result.retryable)})`,
        true,
      );
      const summary = describePullFailure(result);
      if (summary) {
        setWizardMessage(summary, true);
      }
      setWizardMessage(truncateText(result.output), true);
      if (result.retryable || pullState.pending.length > 0) {
        setWizardMessage("Use Retry Failed or Resume Remaining after fixing the issue.", true);
      }
      return false;
    }
    setWizardMessage("All models in this pull plan completed successfully.", true);
    return true;
  } catch (err) {
    setWizardMessage(`Model pull flow failed: ${err}`, true);
    return false;
  } finally {
    pullState.inProgress = false;
    updateWizardButtons();
  }
}

async function pullSelectedModels() {
  const embedModel = setupFormValue(embedModelInput?.value);
  const chatModel = setupFormValue(chatModelInput?.value);
  const models = uniqueStrings([embedModel, chatModel]);
  if (!embedModel || !chatModel) {
    setWizardMessage("Both model names are required before pulling models.");
    return;
  }

  pullState.completed = [];
  pullState.failed = [];
  pullState.pending = [...models];
  updateWizardButtons();

  try {
    await saveSetupConfig(false);
    const success = await executePullPlan(models, "Starting model download");
    if (success) {
      await saveSetupConfig(true);
      setWizardMessage("Setup step complete: selected models are available locally.", true);
      await checkOllamaStatus();
    }
  } catch (err) {
    setWizardMessage(`Model pull flow failed: ${err}`, true);
  }
}

async function retryFailedModels() {
  const failedModels = uniqueStrings(pullState.failed.map((entry) => entry.model));
  if (failedModels.length === 0) {
    setWizardMessage("No failed models to retry.");
    return;
  }
  try {
    await saveSetupConfig(false);
    const success = await executePullPlan(failedModels, "Retrying failed models");
    if (success && pullState.pending.length === 0) {
      await saveSetupConfig(true);
      await checkOllamaStatus();
    }
  } catch (err) {
    setWizardMessage(`Retry failed: ${err}`, true);
  }
}

async function resumeRemainingModels() {
  const remaining = uniqueStrings(pullState.pending);
  if (remaining.length === 0) {
    setWizardMessage("No remaining models to resume.");
    return;
  }
  try {
    await saveSetupConfig(false);
    const success = await executePullPlan(remaining, "Resuming remaining models");
    if (success && pullState.failed.length === 0 && pullState.pending.length === 0) {
      await saveSetupConfig(true);
      await checkOllamaStatus();
    }
  } catch (err) {
    setWizardMessage(`Resume failed: ${err}`, true);
  }
}

async function pickExportFolder() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setIngestMessage("Cannot pick folder: Tauri invoke is unavailable.");
    return;
  }

  try {
    const selected = await invoke("pick_export_folder");
    const normalizedPath = setupFormValue(
      typeof selected === "string" ? selected : selected?.path || "",
    );
    if (!normalizedPath) {
      setIngestMessage("Folder selection cancelled.");
      return;
    }
    setupState = { ...setupState, notes_export_dir: normalizedPath };
    updateExportFolderDisplay();
    await saveExportFolder(normalizedPath);
    setIngestMessage(`Selected folder: ${setupState.notes_export_dir || normalizedPath}`);
  } catch (err) {
    setIngestMessage(`Folder picker failed: ${err}`);
  }
}

async function validateExportFolder() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setIngestMessage("Cannot validate folder: Tauri invoke is unavailable.");
    return null;
  }
  let folder = resolveSelectedExportFolder();
  if (!folder) {
    await loadSetupConfig();
    folder = resolveSelectedExportFolder();
  }
  if (!folder) {
    setIngestMessage("No export folder selected.");
    return null;
  }
  setupState = { ...setupState, notes_export_dir: folder };
  updateExportFolderDisplay();

  try {
    const status = await invoke("validate_export_folder", { path: folder });
    renderExportValidation(status);
    return status;
  } catch (err) {
    setIngestMessage(`Folder validation failed: ${err}`);
    return null;
  }
}

async function runIngest(reindex) {
  let folder = resolveSelectedExportFolder();
  if (!folder) {
    await loadSetupConfig();
    folder = resolveSelectedExportFolder();
  }
  if (!folder) {
    setIngestMessage("No export folder selected.");
    return;
  }
  setupState = { ...setupState, notes_export_dir: folder };
  updateExportFolderDisplay();
  ingestInProgress = true;
  updateFolderActionButtons();
  stopHealthPolling();

  try {
    const validation = await validateExportFolder();
    if (!validation || !validation.exists || !validation.is_dir || validation.supported_files === 0) {
      setIngestMessage("Cannot ingest until folder validation passes.", true);
      return;
    }
    if (validation.permission_denied) {
      setIngestMessage(
        "Cannot ingest because folder access is denied. Grant access and retry validation first.",
        true,
      );
      return;
    }

    const modeLabel = reindex ? "full reindex" : "delta sync";
    const mode = reindex ? "full" : "delta";
    setIngestMessage(`Running ${modeLabel}...`, true);

    const invoke = tauriInvoke();
    const result =
      typeof invoke === "function"
        ? await invoke("backend_ingest", {
            exportDir: folder,
            export_dir: folder,
            mode,
            reindex,
          })
        : await postJson(`${baseUrl}/ingest`, {
            export_dir: folder,
            mode,
            reindex,
          });

    const summary = result?.delta_summary || {};
    const lines = [
      `Ingest completed (${result?.mode || mode}).`,
      `Changed notes: ${result?.changed_notes ?? 0}`,
      `Removed notes: ${result?.removed_notes ?? 0}`,
      `Chunks indexed: ${result?.chunks_indexed ?? 0}`,
      `Added/Updated/Unchanged: ${summary.added_notes ?? 0}/${summary.updated_notes ?? 0}/${summary.unchanged_notes ?? 0}`,
      `Scanned notes: ${summary.scanned_notes ?? 0}`,
      `Last sync time: ${result?.last_sync_time || "n/a"}`,
    ];
    setIngestMessage(lines.join("\n"), true);
    await refreshStats();
  } catch (err) {
    setIngestMessage(`Ingest failed: ${err}`, true);
  } finally {
    ingestInProgress = false;
    updateFolderActionButtons();
    startHealthPolling();
  }
}

function renderSearchResults(items) {
  searchResults.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "backend-url";
    empty.textContent = "No search results.";
    searchResults.appendChild(empty);
    return;
  }

  items.forEach((item, index) => {
    const noteId = item.note_id || item.source?.note_id || "unknown note";
    const scoreText =
      typeof item.score === "number"
        ? `score=${item.score.toFixed(3)} · note=${noteId}`
        : `score=n/a · note=${noteId}`;
    const card = createCard(
      `Result ${index + 1} · ${item.chunk_id || "unknown chunk"}`,
      scoreText,
      item.text || "",
    );
    const actionRow = document.createElement("div");
    actionRow.className = "wizard-actions";
    const askAboutBtn = document.createElement("button");
    askAboutBtn.className = "secondary";
    askAboutBtn.textContent = "Ask About This";
    askAboutBtn.addEventListener("click", () => {
      askQueryInput.value = `Based on this note chunk, answer the question:\n\n${item.text || ""}`;
      switchTab("ask");
    });
    actionRow.appendChild(askAboutBtn);
    card.appendChild(actionRow);
    searchResults.appendChild(card);
  });
}

async function runSearch() {
  const query = setupFormValue(searchQueryInput?.value);
  const topKRaw = setupFormValue(searchTopKInput?.value);
  const topK = Number.parseInt(topKRaw || "5", 10);
  if (!query) {
    searchResults.innerHTML = "";
    searchResults.appendChild(createCard("Validation", null, "Search query must not be empty."));
    return;
  }

  runSearchBtn.disabled = true;
  try {
    const response = await postJson(`${baseUrl}/search`, {
      query,
      top_k: Number.isFinite(topK) && topK > 0 ? topK : 5,
    });
    renderSearchResults(response.results || []);
  } catch (err) {
    searchResults.innerHTML = "";
    searchResults.appendChild(createCard("Search failed", null, String(err)));
  } finally {
    runSearchBtn.disabled = false;
  }
}

function renderCitations(citations) {
  askCitations.innerHTML = "";
  if (!Array.isArray(citations) || citations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "backend-url";
    empty.textContent = "No citations returned.";
    askCitations.appendChild(empty);
    return;
  }

  citations.forEach((citation, index) => {
    const noteId = citation.note_id || citation.source?.note_id || "unknown note";
    const scoreText =
      typeof citation.score === "number"
        ? `score=${citation.score.toFixed(3)} · note=${noteId}`
        : `score=n/a · note=${noteId}`;
    const card = createCard(
      `Citation ${index + 1} · ${citation.chunk_id || "unknown chunk"}`,
      scoreText,
      citation.text || "",
    );
    askCitations.appendChild(card);
  });
}

async function runAsk() {
  const query = setupFormValue(askQueryInput?.value);
  if (!query) {
    askAnswer.textContent = "Question must not be empty.";
    askConfidence.textContent = "n/a";
    askCitations.innerHTML = "";
    return;
  }

  runAskBtn.disabled = true;
  try {
    const response = await postJson(`${baseUrl}/ask`, { query });
    askAnswer.textContent = response.answer || "";
    askConfidence.textContent =
      typeof response.confidence === "number" ? response.confidence.toFixed(3) : "n/a";
    renderCitations(response.citations || []);
  } catch (err) {
    askAnswer.textContent = `Ask failed: ${err}`;
    askConfidence.textContent = "n/a";
    askCitations.innerHTML = "";
  } finally {
    runAskBtn.disabled = false;
  }
}

function wireEvents() {
  refreshHealthBtn?.addEventListener("click", fetchHealth);
  refreshStatsBtn?.addEventListener("click", refreshStats);

  checkOllamaBtn?.addEventListener("click", checkOllamaStatus);
  installOllamaBtn?.addEventListener("click", installOllama);
  startOllamaBtn?.addEventListener("click", startOllama);
  saveSetupBtn?.addEventListener("click", async () => {
    try {
      await saveSetupConfig(false);
    } catch (err) {
      setWizardMessage(`Failed to save setup config: ${err}`);
    }
  });
  pullModelsBtn?.addEventListener("click", pullSelectedModels);
  retryFailedBtn?.addEventListener("click", retryFailedModels);
  resumePullBtn?.addEventListener("click", resumeRemainingModels);

  pickFolderBtn?.addEventListener("click", pickExportFolder);
  validateFolderBtn?.addEventListener("click", validateExportFolder);
  ingestFullBtn?.addEventListener("click", () => runIngest(true));
  syncNowBtn?.addEventListener("click", () => runIngest(false));

  runSearchBtn?.addEventListener("click", runSearch);
  runAskBtn?.addEventListener("click", runAsk);

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      switchTab(tab.dataset.tabTarget);
    });
  });
}

async function start() {
  baseUrl = await resolveBackendUrl();
  healthUrl = `${baseUrl}/health`;
  if (backendUrlText) {
    backendUrlText.textContent = baseUrl;
  }

  wireEvents();
  await fetchHealth();
  startHealthPolling();

  await loadSetupConfig();
  await checkOllamaStatus();
  await refreshStats();
  updateWizardButtons();
  updateFolderActionButtons();
}

start();
