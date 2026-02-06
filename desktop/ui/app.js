const HEALTH_URL = "http://127.0.0.1:8001/health";
const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const output = document.getElementById("health-output");
const refreshBtn = document.getElementById("refresh");

async function fetchHealth() {
  statusText.textContent = "Checking...";
  statusDot.style.background = "#f0b429";

  try {
    const res = await fetch(HEALTH_URL);
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
fetchHealth();
setInterval(fetchHealth, 5000);
