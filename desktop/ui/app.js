const DEFAULT_BACKEND_URL = "http://127.0.0.1:8001";
const DEFAULT_SETUP_CONFIG = {
  embed_model: "nomic-embed-text",
  chat_model: "neural-chat",
  wizard_completed: false,
};
const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const backendUrlText = document.getElementById("backend-url");
const output = document.getElementById("health-output");
const refreshBtn = document.getElementById("refresh");
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
let healthUrl = `${DEFAULT_BACKEND_URL}/health`;
const pullState = {
  inProgress: false,
  completed: [],
  failed: [],
  pending: [],
  activePlan: [],
};
let wizardAvailable = true;

function tauriInvoke() {
  return window.__TAURI__?.tauri?.invoke;
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

function getCurrentSetupConfig() {
  return {
    embed_model: setupFormValue(embedModelInput?.value),
    chat_model: setupFormValue(chatModelInput?.value),
    wizard_completed: false,
  };
}

function setWizardMessage(message, append = false) {
  if (!wizardOutput) {
    return;
  }
  wizardOutput.textContent = append
    ? `${wizardOutput.textContent}\n${message}`
    : message;
}

function setSetupForm(config) {
  if (embedModelInput) {
    embedModelInput.value = config.embed_model || DEFAULT_SETUP_CONFIG.embed_model;
  }
  if (chatModelInput) {
    chatModelInput.value = config.chat_model || DEFAULT_SETUP_CONFIG.chat_model;
  }
}

function setWizardButtonsEnabled(enabled) {
  wizardAvailable = enabled;
  updatePullActionButtons();
}

function uniqueStrings(values) {
  const outputValues = [];
  values.forEach((value) => {
    if (value && !outputValues.includes(value)) {
      outputValues.push(value);
    }
  });
  return outputValues;
}

function addUnique(array, value) {
  if (value && !array.includes(value)) {
    array.push(value);
  }
}

function updatePullActionButtons() {
  const disableBecauseUnavailable = !wizardAvailable;
  const disableBecauseBusy = pullState.inProgress;
  const disableBase = disableBecauseUnavailable || disableBecauseBusy;

  [checkOllamaBtn, installOllamaBtn, startOllamaBtn, saveSetupBtn, pullModelsBtn].forEach((button) => {
    if (button) {
      button.disabled = disableBase;
    }
  });

  if (pullModelsBtn) {
    pullModelsBtn.disabled = disableBase;
  }
  if (saveSetupBtn) {
    saveSetupBtn.disabled = disableBase;
  }
  if (retryFailedBtn) {
    retryFailedBtn.disabled = disableBase || pullState.failed.length === 0;
  }
  if (resumePullBtn) {
    resumePullBtn.disabled = disableBase || pullState.pending.length === 0;
  }
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

async function resolveBackendUrl() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    return DEFAULT_BACKEND_URL;
  }

  try {
    const baseUrl = await invoke("backend_base_url");
    if (typeof baseUrl === "string" && baseUrl.trim().length > 0) {
      return baseUrl;
    }
  } catch (err) {
    output.textContent = `Could not resolve backend URL from app shell: ${err}`;
  }

  return DEFAULT_BACKEND_URL;
}

async function fetchHealth() {
  statusText.textContent = "Checking...";
  statusDot.style.background = "#f0b429";

  try {
    const res = await fetch(healthUrl);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
    statusText.textContent = data.status === "ok" ? "Healthy" : "Degraded";
    statusDot.style.background = data.status === "ok" ? "#2f9e44" : "#f59f00";
  } catch (err) {
    output.textContent = `Health check failed: ${err}`;
    statusText.textContent = "Offline";
    statusDot.style.background = "#d9480f";
  }
}

refreshBtn.addEventListener("click", fetchHealth);

async function loadSetupConfig() {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    setSetupForm(DEFAULT_SETUP_CONFIG);
    setWizardButtonsEnabled(false);
    setWizardMessage("Setup wizard actions require the Tauri desktop shell.");
    return DEFAULT_SETUP_CONFIG;
  }

  try {
    const loaded = await invoke("setup_load_config");
    const config = {
      embed_model: loaded?.embed_model || DEFAULT_SETUP_CONFIG.embed_model,
      chat_model: loaded?.chat_model || DEFAULT_SETUP_CONFIG.chat_model,
      wizard_completed: Boolean(loaded?.wizard_completed),
    };
    setSetupForm(config);
    setWizardMessage(
      config.wizard_completed
        ? "Setup config loaded. Wizard already marked complete."
        : "Setup config loaded. Run Ollama checks and pull models.",
    );
    return config;
  } catch (err) {
    setSetupForm(DEFAULT_SETUP_CONFIG);
    setWizardMessage(`Could not load setup config: ${err}`);
    return DEFAULT_SETUP_CONFIG;
  }
}

async function saveSetupConfig(wizardCompleted) {
  const invoke = tauriInvoke();
  if (typeof invoke !== "function") {
    throw new Error("Tauri invoke is not available");
  }

  const config = getCurrentSetupConfig();
  if (!config.embed_model || !config.chat_model) {
    throw new Error("Both embedding and chat model names are required");
  }

  const saved = await invoke("setup_save_config", {
    embed_model: config.embed_model,
    chat_model: config.chat_model,
    wizard_completed: wizardCompleted,
  });
  setWizardMessage(
    `Saved setup config (embed=${saved.embed_model}, chat=${saved.chat_model}, completed=${saved.wizard_completed})`,
  );
  return saved;
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

function uniqueModels(embedModel, chatModel) {
  return uniqueStrings([embedModel, chatModel]);
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
      return "Ollama is not installed or not available in PATH.";
    default:
      return "Unknown pull error.";
  }
}

function failuresForModels(models) {
  const modelSet = new Set(models);
  return pullState.failed.filter((entry) => modelSet.has(entry.model));
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

  const planSet = new Set(plan);
  pullState.failed = pullState.failed.filter((entry) => !planSet.has(entry.model));
  pullState.pending = [...plan];
  pullState.activePlan = [...plan];

  pullState.inProgress = true;
  updatePullActionButtons();
  try {
    setWizardMessage(`${label}: ${plan.join(", ")}`, true);

    for (let idx = 0; idx < plan.length; idx += 1) {
      const model = plan[idx];
      setWizardMessage(`Pulling ${model}...`, true);
      const result = await invoke("ollama_pull_model", { model });
      pullState.pending = plan.slice(idx + 1);

      if (result.success) {
        addUnique(pullState.completed, model);
        setWizardMessage(`Model ${model}: success`, true);
        continue;
      }

      pullState.failed = pullState.failed.filter((entry) => entry.model !== model);
      pullState.failed.push({
        model,
        error_type: result.error_type || "unknown",
        retryable: Boolean(result.retryable),
      });

      const failureSummary = describePullFailure(result);
      setWizardMessage(
        `Model ${model}: failed (${result.error_type || "unknown"}, retryable=${Boolean(result.retryable)})`,
        true,
      );
      if (failureSummary) {
        setWizardMessage(failureSummary, true);
      }
      setWizardMessage(truncateText(result.output), true);
      if (result.retryable || pullState.pending.length > 0) {
        setWizardMessage("Use 'Retry Failed' or 'Resume Remaining' after fixing the issue.", true);
      }
      return false;
    }

    if (failuresForModels(plan).length === 0) {
      setWizardMessage("All models in this pull plan completed successfully.", true);
      return true;
    }
    return false;
  } catch (err) {
    setWizardMessage(`Model pull flow failed: ${err}`, true);
    return false;
  } finally {
    pullState.inProgress = false;
    updatePullActionButtons();
  }
}

async function pullSelectedModels() {
  const embedModel = setupFormValue(embedModelInput?.value);
  const chatModel = setupFormValue(chatModelInput?.value);
  const models = uniqueModels(embedModel, chatModel);
  if (!embedModel || !chatModel) {
    setWizardMessage("Both model names are required before pulling models.");
    return;
  }

  pullState.completed = [];
  pullState.failed = [];
  pullState.pending = [...models];
  pullState.activePlan = [...models];
  updatePullActionButtons();

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
      setWizardMessage("Failed models recovered successfully.", true);
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
      setWizardMessage("Remaining models completed successfully.", true);
      await checkOllamaStatus();
    }
  } catch (err) {
    setWizardMessage(`Resume failed: ${err}`, true);
  }
}

if (checkOllamaBtn) {
  checkOllamaBtn.addEventListener("click", checkOllamaStatus);
}
if (installOllamaBtn) {
  installOllamaBtn.addEventListener("click", installOllama);
}
if (startOllamaBtn) {
  startOllamaBtn.addEventListener("click", startOllama);
}
if (saveSetupBtn) {
  saveSetupBtn.addEventListener("click", async () => {
    try {
      await saveSetupConfig(false);
    } catch (err) {
      setWizardMessage(`Failed to save setup config: ${err}`);
    }
  });
}
if (pullModelsBtn) {
  pullModelsBtn.addEventListener("click", pullSelectedModels);
}
if (retryFailedBtn) {
  retryFailedBtn.addEventListener("click", retryFailedModels);
}
if (resumePullBtn) {
  resumePullBtn.addEventListener("click", resumeRemainingModels);
}

async function start() {
  const baseUrl = await resolveBackendUrl();
  healthUrl = `${baseUrl}/health`;

  if (backendUrlText) {
    backendUrlText.textContent = baseUrl;
  }

  await fetchHealth();
  setInterval(fetchHealth, 5000);
  await loadSetupConfig();
  await checkOllamaStatus();
  updatePullActionButtons();
}

start();
