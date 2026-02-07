const DEFAULT_BACKEND_URL = "http://127.0.0.1:8001";
const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const backendUrlText = document.getElementById("backend-url");
const output = document.getElementById("health-output");
const refreshBtn = document.getElementById("refresh");
let healthUrl = `${DEFAULT_BACKEND_URL}/health`;

async function resolveBackendUrl() {
  const tauriInvoke = window.__TAURI__?.tauri?.invoke;
  if (typeof tauriInvoke !== "function") {
    return DEFAULT_BACKEND_URL;
  }

  try {
    const baseUrl = await tauriInvoke("backend_base_url");
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

async function start() {
  const baseUrl = await resolveBackendUrl();
  healthUrl = `${baseUrl}/health`;

  if (backendUrlText) {
    backendUrlText.textContent = baseUrl;
  }

  await fetchHealth();
  setInterval(fetchHealth, 5000);
}

start();
