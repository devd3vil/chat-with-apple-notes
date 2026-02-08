#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, ErrorKind};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{Manager, State};

const DEFAULT_BACKEND_PORT: u16 = 8001;
const PORT_SCAN_SIZE: u16 = 100;
const HEALTH_CHECK_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_MS: u64 = 250;
const BACKEND_INGEST_TIMEOUT_SECS: u64 = 7200;
const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const SETUP_CONFIG_FILE: &str = "setup.json";
const CHAT_THREADS_FILE: &str = "chat_threads.json";
const APP_OWNED_EXPORT_SUBDIR: &str = "NotesLensExport";
const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
const DEFAULT_CHAT_MODEL: &str = "neural-chat";
const DEFAULT_MODEL_PULL_TIMEOUT_SECS: u64 = 1800;

struct BackendRuntime {
    child: Option<Child>,
    port: u16,
}

struct BackendState {
    runtime: Mutex<BackendRuntime>,
}

struct ExportJobState {
    runtime: Arc<Mutex<ExportJobRuntime>>,
}

#[derive(Debug, Default)]
struct ExportJobRuntime {
    snapshot: Option<ExportJobSnapshot>,
    cancel_requested: bool,
    process_pid: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppConfig {
    has_completed_onboarding: bool,
    export_folder_path: Option<String>,
    last_synced_at: Option<String>,
    embed_model: String,
    chat_model: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            has_completed_onboarding: false,
            export_folder_path: None,
            last_synced_at: None,
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            chat_model: DEFAULT_CHAT_MODEL.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SetupConfig {
    embed_model: String,
    chat_model: String,
    notes_export_dir: Option<String>,
    wizard_completed: bool,
}

impl Default for SetupConfig {
    fn default() -> Self {
        Self {
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            chat_model: DEFAULT_CHAT_MODEL.to_string(),
            notes_export_dir: None,
            wizard_completed: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct OllamaStatus {
    installed: bool,
    running: bool,
    version: Option<String>,
    models: Vec<String>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct PullModelResult {
    model: String,
    success: bool,
    output: String,
    error_type: String,
    retryable: bool,
    timed_out: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ExportFolderStatus {
    path: String,
    exists: bool,
    is_dir: bool,
    supported_files: usize,
    permission_denied: bool,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    id: String,
    role: String,
    text: String,
    timestamp: String,
    citations: Option<Vec<serde_json::Value>>,
    confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatThread {
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportJobSnapshot {
    job_id: String,
    status: String,
    phase: String,
    current: usize,
    total: Option<usize>,
    message: String,
    started_at: String,
    finished_at: Option<String>,
    target_dir: String,
    error: Option<String>,
    logs: String,
}

fn find_backend_workdir() -> PathBuf {
    if let Ok(dir) = std::env::var("BACKEND_WORKDIR") {
        return PathBuf::from(dir);
    }

    let current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if current.join("main.py").exists() {
        return current;
    }

    let candidate = current.join("../..");
    if candidate.join("main.py").exists() {
        return candidate;
    }

    current
}

fn parse_backend_port(raw: Option<String>) -> Result<Option<u16>, String> {
    let Some(port_raw) = raw else {
        return Ok(None);
    };

    let parsed = port_raw
        .trim()
        .parse::<u16>()
        .map_err(|_| format!("Invalid BACKEND_PORT value: {port_raw}"))?;
    if parsed == 0 {
        return Err("Invalid BACKEND_PORT value: 0".to_string());
    }
    Ok(Some(parsed))
}

fn is_port_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn select_backend_port() -> Result<u16, String> {
    let requested = parse_backend_port(std::env::var("BACKEND_PORT").ok())?;
    if let Some(port) = requested {
        if is_port_available(port) {
            return Ok(port);
        }
        return Err(format!(
            "Requested BACKEND_PORT {port} is not available on 127.0.0.1"
        ));
    }

    let start = u32::from(DEFAULT_BACKEND_PORT);
    let end = start + u32::from(PORT_SCAN_SIZE);
    for candidate in start..end {
        let port = candidate as u16;
        if is_port_available(port) {
            return Ok(port);
        }
    }

    Err(format!(
        "No available backend port found in range {DEFAULT_BACKEND_PORT}-{}",
        DEFAULT_BACKEND_PORT + PORT_SCAN_SIZE - 1
    ))
}

fn spawn_backend(port: u16, app_config: Option<&AppConfig>) -> Result<Child, String> {
    let backend_cmd = std::env::var("BACKEND_CMD").unwrap_or_else(|_| "python".to_string());
    let mut args = vec![
        "-m".to_string(),
        "uvicorn".to_string(),
        "main:app".to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
    ];

    if let Ok(extra) = std::env::var("BACKEND_ARGS") {
        if !extra.trim().is_empty() {
            args.extend(extra.split_whitespace().map(|s| s.to_string()));
        }
    }

    let workdir = find_backend_workdir();

    let mut command = Command::new(backend_cmd);
    command
        .args(args)
        .current_dir(workdir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    for (key, value) in backend_env_from_config(app_config) {
        command.env(key, value);
    }

    command
        .spawn()
        .map_err(|err| format!("Failed to start backend: {err}"))
}

fn backend_env_from_config(config: Option<&AppConfig>) -> Vec<(String, String)> {
    let mut envs = Vec::new();
    let Some(config) = config else {
        return envs;
    };

    let embed_model = config.embed_model.trim();
    if !embed_model.is_empty() {
        envs.push((
            "OLLAMA_EMBEDDING_MODEL".to_string(),
            embed_model.to_string(),
        ));
    }

    let chat_model = config.chat_model.trim();
    if !chat_model.is_empty() {
        envs.push(("OLLAMA_CHAT_MODEL".to_string(), chat_model.to_string()));
    }

    if let Some(folder) = config.export_folder_path.as_deref() {
        if let Ok(owned) = app_owned_export_subfolder(folder) {
            envs.push((
                "NOTES_EXPORT_DIR".to_string(),
                owned.to_string_lossy().to_string(),
            ));
        }
    }

    envs
}

fn wait_for_backend_health(port: u16, child: &mut Child) -> Result<(), String> {
    let health_url = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + Duration::from_secs(HEALTH_CHECK_TIMEOUT_SECS);

    loop {
        let current_error = match ureq::get(&health_url)
            .timeout(Duration::from_secs(2))
            .call()
        {
            Ok(response) if response.status() == 200 => return Ok(()),
            Ok(response) => format!("Health endpoint returned HTTP {}", response.status()),
            Err(err) => err.to_string(),
        };

        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "Backend process exited before health was ready: {status}"
            ));
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "Backend health check timed out after {HEALTH_CHECK_TIMEOUT_SECS}s at {health_url}: {current_error}"
            ));
        }

        thread::sleep(Duration::from_millis(HEALTH_CHECK_INTERVAL_MS));
    }
}

fn app_config_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let config_dir = app
        .path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Unable to resolve app config directory".to_string())?;
    fs::create_dir_all(&config_dir)
        .map_err(|err| format!("Failed to create config dir {config_dir:?}: {err}"))?;
    Ok(config_dir.join(SETUP_CONFIG_FILE))
}

fn chat_threads_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let config_dir = app
        .path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Unable to resolve app config directory".to_string())?;
    fs::create_dir_all(&config_dir)
        .map_err(|err| format!("Failed to create config dir {config_dir:?}: {err}"))?;
    Ok(config_dir.join(CHAT_THREADS_FILE))
}

fn normalize_string(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn normalize_bool(value: Option<&serde_json::Value>) -> Option<bool> {
    value.and_then(|v| v.as_bool())
}

fn normalize_app_config(value: &serde_json::Value) -> AppConfig {
    let has_completed_onboarding = normalize_bool(
        value
            .get("has_completed_onboarding")
            .or_else(|| value.get("hasCompletedOnboarding"))
            .or_else(|| value.get("wizard_completed"))
            .or_else(|| value.get("wizardCompleted")),
    )
    .unwrap_or(false);

    let export_folder_path = normalize_string(
        value
            .get("export_folder_path")
            .or_else(|| value.get("exportFolderPath"))
            .or_else(|| value.get("notes_export_dir"))
            .or_else(|| value.get("notesExportDir")),
    );

    let last_synced_at = normalize_string(
        value
            .get("last_synced_at")
            .or_else(|| value.get("lastSyncedAt"))
            .or_else(|| value.get("last_sync_timestamp"))
            .or_else(|| value.get("lastSyncTimestamp")),
    );

    let embed_model =
        normalize_string(value.get("embed_model").or_else(|| value.get("embedModel")))
            .unwrap_or_else(|| DEFAULT_EMBED_MODEL.to_string());

    let chat_model = normalize_string(value.get("chat_model").or_else(|| value.get("chatModel")))
        .unwrap_or_else(|| DEFAULT_CHAT_MODEL.to_string());

    AppConfig {
        has_completed_onboarding,
        export_folder_path,
        last_synced_at,
        embed_model,
        chat_model,
    }
}

fn sanitize_app_config(config: AppConfig) -> AppConfig {
    AppConfig {
        has_completed_onboarding: config.has_completed_onboarding,
        export_folder_path: config
            .export_folder_path
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
        last_synced_at: config
            .last_synced_at
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
        embed_model: {
            let trimmed = config.embed_model.trim().to_string();
            if trimmed.is_empty() {
                DEFAULT_EMBED_MODEL.to_string()
            } else {
                trimmed
            }
        },
        chat_model: {
            let trimmed = config.chat_model.trim().to_string();
            if trimmed.is_empty() {
                DEFAULT_CHAT_MODEL.to_string()
            } else {
                trimmed
            }
        },
    }
}

fn read_app_config(path: &Path) -> Result<AppConfig, String> {
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let raw = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read app config {path:?}: {err}"))?;

    if let Ok(parsed) = serde_json::from_str::<AppConfig>(&raw) {
        return Ok(sanitize_app_config(parsed));
    }

    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("Invalid app config JSON at {path:?}: {err}"))?;
    Ok(sanitize_app_config(normalize_app_config(&value)))
}

fn write_app_config(path: &Path, config: &AppConfig) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(config)
        .map_err(|err| format!("Failed to serialize app config: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("Failed to write app config {path:?}: {err}"))
}

#[allow(dead_code)]
fn requires_onboarding(config: &AppConfig) -> bool {
    !config.has_completed_onboarding
        || config
            .export_folder_path
            .as_ref()
            .map(|value| value.trim().is_empty())
            .unwrap_or(true)
        || config
            .last_synced_at
            .as_ref()
            .map(|value| value.trim().is_empty())
            .unwrap_or(true)
}

fn setup_config_from_app(config: &AppConfig) -> SetupConfig {
    SetupConfig {
        embed_model: config.embed_model.clone(),
        chat_model: config.chat_model.clone(),
        notes_export_dir: config.export_folder_path.clone(),
        wizard_completed: config.has_completed_onboarding,
    }
}

fn parse_ollama_list_models(raw: &str) -> Vec<String> {
    let mut models = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("NAME") {
            continue;
        }
        if let Some(name) = trimmed.split_whitespace().next() {
            if !models.iter().any(|model| model == name) {
                models.push(name.to_string());
            }
        }
    }
    models
}

fn is_supported_export_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            matches!(ext.as_str(), "html" | "htm" | "md" | "markdown" | "txt")
        })
        .unwrap_or(false)
}

fn count_supported_files(root: &Path) -> (usize, bool) {
    let mut count = 0usize;
    let mut permission_denied = false;
    let mut stack = vec![root.to_path_buf()];

    while let Some(current) = stack.pop() {
        let entries = match fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(err) => {
                if err.kind() == ErrorKind::PermissionDenied {
                    permission_denied = true;
                }
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    if err.kind() == ErrorKind::PermissionDenied {
                        permission_denied = true;
                    }
                    continue;
                }
            };
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_supported_export_file(&path) {
                count += 1;
            }
        }
    }

    (count, permission_denied)
}

fn collect_supported_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !root.exists() {
        return files;
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_supported_export_file(&path) {
                files.push(path);
            }
        }
    }

    files.sort_by_key(|path| path.to_string_lossy().to_string());
    files
}

fn format_export_log_label(root: &Path, file_path: &Path) -> String {
    let relative = file_path.strip_prefix(root).unwrap_or(file_path);
    relative.to_string_lossy().replace('\\', "/")
}

fn classify_pull_failure(output: &str, timed_out: bool) -> (String, bool) {
    if timed_out {
        return ("timeout".to_string(), true);
    }

    let lower = output.to_lowercase();
    if lower.contains("no space left")
        || lower.contains("disk full")
        || lower.contains("not enough space")
    {
        return ("disk_full".to_string(), false);
    }
    if lower.contains("connection refused")
        || lower.contains("connection reset")
        || lower.contains("i/o timeout")
        || lower.contains("network is unreachable")
        || lower.contains("temporary failure in name resolution")
    {
        return ("offline".to_string(), true);
    }
    if lower.contains("not found")
        || lower.contains("command not found")
        || lower.contains("failed to run `ollama`")
    {
        return ("not_installed".to_string(), false);
    }
    ("unknown".to_string(), true)
}

fn model_pull_timeout() -> Duration {
    let timeout = std::env::var("OLLAMA_PULL_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MODEL_PULL_TIMEOUT_SECS);
    Duration::from_secs(timeout)
}

fn is_ollama_running() -> bool {
    let tags_url = format!("{OLLAMA_BASE_URL}/api/tags");
    ureq::get(&tags_url)
        .timeout(Duration::from_secs(2))
        .call()
        .map(|response| response.status() == 200)
        .unwrap_or(false)
}

fn run_command_capture(command: &str, args: &[&str]) -> Result<(bool, String), String> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|err| format!("Failed to run `{command}`: {err}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let combined = match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (false, true) => stdout,
        (true, false) => stderr,
        (false, false) => format!("{stdout}\n{stderr}"),
    };
    Ok((output.status.success(), combined))
}

fn run_command_capture_with_timeout(
    command: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<(bool, String, bool), String> {
    let mut child = Command::new(command)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("Failed to run `{command}`: {err}"))?;

    let start = Instant::now();
    let timed_out = loop {
        let maybe_status = child
            .try_wait()
            .map_err(|err| format!("Failed to poll `{command}` process: {err}"))?;
        if maybe_status.is_some() {
            break false;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            break true;
        }
        thread::sleep(Duration::from_millis(200));
    };

    let output = child
        .wait_with_output()
        .map_err(|err| format!("Failed to collect command output for `{command}`: {err}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let combined = match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => "(no output)".to_string(),
        (false, true) => stdout,
        (true, false) => stderr,
        (false, false) => format!("{stdout}\n{stderr}"),
    };

    Ok((output.status.success() && !timed_out, combined, timed_out))
}

fn now_timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn notes_export_script_path() -> PathBuf {
    find_backend_workdir()
        .join("scripts")
        .join("export_notes.applescript")
}

fn app_owned_export_subfolder(base: &str) -> Result<PathBuf, String> {
    let trimmed = base.trim();
    if trimmed.is_empty() {
        return Err("export_folder_path must not be empty".to_string());
    }
    Ok(Path::new(trimmed).join(APP_OWNED_EXPORT_SUBDIR))
}

fn build_export_script_args(
    script_path: &Path,
    target_dir: &Path,
    max_files: Option<usize>,
) -> Vec<String> {
    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        target_dir.to_string_lossy().to_string(),
    ];
    if let Some(limit) = max_files.filter(|value| *value > 0) {
        args.push(limit.to_string());
    }
    args
}

fn push_log_line(existing: &str, new_line: &str, max_lines: usize) -> String {
    let trimmed = new_line.trim();
    if trimmed.is_empty() {
        return existing.to_string();
    }
    let mut lines: Vec<String> = existing
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    lines.push(trimmed.to_string());
    if lines.len() > max_lines {
        let keep_from = lines.len() - max_lines;
        lines = lines.split_off(keep_from);
    }
    lines.join("\n")
}

fn update_snapshot_from_progress_line(
    snapshot: &mut ExportJobSnapshot,
    line: &str,
) -> Option<String> {
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(line) else {
        return None;
    };

    let Some(phase) = payload.get("phase").and_then(|value| value.as_str()) else {
        return None;
    };
    if phase != "export" {
        return None;
    }

    if let Some(total) = payload.get("total").and_then(|value| value.as_u64()) {
        snapshot.total = Some(total as usize);
    }
    if let Some(current) = payload.get("exported").and_then(|value| value.as_u64()) {
        snapshot.current = current as usize;
    }

    let type_name = payload
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("progress");
    let note_name = payload
        .get("note")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if type_name == "complete" {
        let total = snapshot.total.unwrap_or(snapshot.current);
        snapshot.current = total;
        snapshot.message = format!("Export complete. Exported {} of {} files.", total, total);
        return Some("Export complete.".to_string());
    } else if let Some(total) = snapshot.total {
        snapshot.message = format!("Pulling notes... {} of {} files", snapshot.current, total);
    } else {
        snapshot.message = format!("Pulling notes... {} files", snapshot.current);
    }

    if let Some(note) = note_name {
        if let Some(total) = snapshot.total {
            return Some(format!("{} / {} - {}", snapshot.current, total, note));
        }
        return Some(format!("{} - {}", snapshot.current, note));
    }

    None
}

fn start_stream_reader<R: std::io::Read + Send + 'static>(
    stream: R,
    is_stderr: bool,
    sender: mpsc::Sender<(bool, String)>,
) {
    thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader.read_line(&mut line).unwrap_or(0);
            if read == 0 {
                break;
            }
            let _ = sender.send((is_stderr, line.trim_end().to_string()));
        }
    });
}

fn prune_chat_threads(mut threads: Vec<ChatThread>) -> Vec<ChatThread> {
    threads.sort_by(|a, b| {
        parse_timestamp_for_sort(&b.updated_at).cmp(&parse_timestamp_for_sort(&a.updated_at))
    });
    threads.truncate(10);
    threads
}

fn parse_timestamp_for_sort(raw: &str) -> u128 {
    raw.trim().parse::<u128>().unwrap_or(0)
}

#[tauri::command]
fn backend_base_url(state: State<BackendState>) -> Result<String, String> {
    let runtime_guard = state.runtime.lock().map_err(|_| "Lock error".to_string())?;
    Ok(format!("http://127.0.0.1:{}", runtime_guard.port))
}

#[tauri::command]
async fn backend_ingest(
    state: State<'_, BackendState>,
    export_dir: String,
    mode: String,
    reindex: bool,
) -> Result<serde_json::Value, String> {
    let export_dir = export_dir.trim().to_string();
    if export_dir.is_empty() {
        return Err("export_dir must not be empty".to_string());
    }

    let mode = mode.trim().to_ascii_lowercase();
    if mode != "full" && mode != "delta" {
        return Err("mode must be either `full` or `delta`".to_string());
    }

    let port = {
        let runtime_guard = state.runtime.lock().map_err(|_| "Lock error".to_string())?;
        runtime_guard.port
    };
    let url = format!("http://127.0.0.1:{port}/ingest");

    tauri::async_runtime::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let payload = serde_json::json!({
            "export_dir": export_dir,
            "mode": mode,
            "reindex": reindex,
        });
        let payload_raw = serde_json::to_string(&payload)
            .map_err(|err| format!("Failed to serialize ingest payload: {err}"))?;

        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(BACKEND_INGEST_TIMEOUT_SECS))
            .send_string(&payload_raw);

        match response {
            Ok(resp) => {
                let body = resp
                    .into_string()
                    .map_err(|err| format!("Failed to read ingest response body: {err}"))?;
                serde_json::from_str::<serde_json::Value>(&body)
                    .map_err(|err| format!("Failed to parse ingest response JSON: {err}"))
            }
            Err(ureq::Error::Status(status, resp)) => {
                let body = resp
                    .into_string()
                    .unwrap_or_else(|_| "(failed to read response body)".to_string());
                Err(format!("HTTP {status}: {body}"))
            }
            Err(err) => Err(format!("Backend ingest request failed: {err}")),
        }
    })
    .await
    .map_err(|err| format!("Ingest task failed: {err}"))?
}

#[tauri::command]
fn load_app_config(app: tauri::AppHandle) -> Result<AppConfig, String> {
    let path = app_config_path(&app)?;
    let config = read_app_config(&path)?;
    write_app_config(&path, &config)?;
    Ok(config)
}

#[tauri::command]
fn save_app_config(
    app: tauri::AppHandle,
    has_completed_onboarding: bool,
    export_folder_path: Option<String>,
    last_synced_at: Option<String>,
    embed_model: String,
    chat_model: String,
) -> Result<AppConfig, String> {
    let path = app_config_path(&app)?;
    let updated = sanitize_app_config(AppConfig {
        has_completed_onboarding,
        export_folder_path,
        last_synced_at,
        embed_model,
        chat_model,
    });
    write_app_config(&path, &updated)?;
    Ok(updated)
}

#[tauri::command]
fn setup_load_config(app: tauri::AppHandle) -> Result<SetupConfig, String> {
    let config = load_app_config(app)?;
    Ok(setup_config_from_app(&config))
}

#[tauri::command]
fn setup_save_config(
    app: tauri::AppHandle,
    embed_model: String,
    chat_model: String,
    wizard_completed: bool,
) -> Result<SetupConfig, String> {
    let path = app_config_path(&app)?;
    let mut config = read_app_config(&path)?;
    config.embed_model = embed_model;
    config.chat_model = chat_model;
    config.has_completed_onboarding = wizard_completed;
    config = sanitize_app_config(config);
    write_app_config(&path, &config)?;
    Ok(setup_config_from_app(&config))
}

#[tauri::command]
fn setup_save_notes_export_dir(
    app: tauri::AppHandle,
    notes_export_dir: Option<String>,
) -> Result<SetupConfig, String> {
    let path = app_config_path(&app)?;
    let mut config = read_app_config(&path)?;
    config.export_folder_path = notes_export_dir;
    config = sanitize_app_config(config);
    write_app_config(&path, &config)?;
    Ok(setup_config_from_app(&config))
}

#[tauri::command]
fn load_chat_threads(app: tauri::AppHandle) -> Result<Vec<ChatThread>, String> {
    let path = chat_threads_path(&app)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read chat store {path:?}: {err}"))?;
    let parsed: Vec<ChatThread> = serde_json::from_str(&raw)
        .map_err(|err| format!("Invalid chat store JSON at {path:?}: {err}"))?;
    Ok(prune_chat_threads(parsed))
}

#[tauri::command]
fn save_chat_threads(
    app: tauri::AppHandle,
    threads: Vec<ChatThread>,
) -> Result<Vec<ChatThread>, String> {
    let path = chat_threads_path(&app)?;
    let pruned = prune_chat_threads(threads);
    let raw = serde_json::to_string_pretty(&pruned)
        .map_err(|err| format!("Failed to serialize chat store: {err}"))?;
    fs::write(&path, raw).map_err(|err| format!("Failed to write chat store {path:?}: {err}"))?;
    Ok(pruned)
}

#[tauri::command]
fn start_export_job(
    export_state: State<ExportJobState>,
    export_folder_path: String,
    max_files: Option<usize>,
) -> Result<ExportJobSnapshot, String> {
    let target_dir = app_owned_export_subfolder(&export_folder_path)?;
    fs::create_dir_all(&target_dir)
        .map_err(|err| format!("Failed to create export folder {target_dir:?}: {err}"))?;

    let mut guard = export_state
        .runtime
        .lock()
        .map_err(|_| "Lock error".to_string())?;

    if let Some(existing) = guard.snapshot.as_ref() {
        if existing.status == "queued"
            || existing.status == "running"
            || existing.status == "cancelling"
        {
            return Err("An export job is already running".to_string());
        }
    }

    let job_id = format!(
        "export-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    let snapshot = ExportJobSnapshot {
        job_id: job_id.clone(),
        status: "queued".to_string(),
        phase: "export".to_string(),
        current: 0,
        total: max_files.filter(|value| *value > 0),
        message: "Export queued.".to_string(),
        started_at: now_timestamp_string(),
        finished_at: None,
        target_dir: target_dir.to_string_lossy().to_string(),
        error: None,
        logs: String::new(),
    };

    guard.cancel_requested = false;
    guard.process_pid = None;
    guard.snapshot = Some(snapshot.clone());
    drop(guard);

    let runtime = Arc::clone(&export_state.runtime);
    let target_dir_for_thread = target_dir.clone();
    let job_id_for_thread = job_id.clone();

    thread::spawn(move || {
        let script_path = notes_export_script_path();
        if !script_path.exists() {
            if let Ok(mut runtime_guard) = runtime.lock() {
                if let Some(s) = runtime_guard.snapshot.as_mut() {
                    if s.job_id == job_id_for_thread {
                        s.status = "failed".to_string();
                        s.message = "Export script not found.".to_string();
                        s.error = Some(format!("Missing script at {}", script_path.display()));
                        s.finished_at = Some(now_timestamp_string());
                    }
                }
            }
            return;
        }

        let args = build_export_script_args(&script_path, &target_dir_for_thread, max_files);
        let mut command = Command::new("osascript");
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                if let Ok(mut runtime_guard) = runtime.lock() {
                    if let Some(s) = runtime_guard.snapshot.as_mut() {
                        if s.job_id == job_id_for_thread {
                            s.status = "failed".to_string();
                            s.message = "Failed to start export script.".to_string();
                            s.error = Some(err.to_string());
                            s.finished_at = Some(now_timestamp_string());
                        }
                    }
                }
                return;
            }
        };

        let (line_sender, line_receiver) = mpsc::channel::<(bool, String)>();
        if let Some(stdout) = child.stdout.take() {
            start_stream_reader(stdout, false, line_sender.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            start_stream_reader(stderr, true, line_sender.clone());
        }
        drop(line_sender);

        let pid = child.id();
        if let Ok(mut runtime_guard) = runtime.lock() {
            runtime_guard.process_pid = Some(pid);
            if let Some(s) = runtime_guard.snapshot.as_mut() {
                if s.job_id == job_id_for_thread {
                    s.status = "running".to_string();
                    s.message = "Pulling notes...".to_string();
                    s.logs = push_log_line(&s.logs, "Starting export...", 10);
                }
            }
        }

        let mut seen_files: HashSet<String> = HashSet::new();
        let mut last_fallback_scan = Instant::now();
        loop {
            while let Ok((is_stderr, line)) = line_receiver.try_recv() {
                if let Ok(mut runtime_guard) = runtime.lock() {
                    if let Some(snapshot) = runtime_guard.snapshot.as_mut() {
                        if snapshot.job_id != job_id_for_thread {
                            continue;
                        }

                        if is_stderr {
                            let prefixed = format!("stderr: {line}");
                            snapshot.logs = push_log_line(&snapshot.logs, &prefixed, 10);
                        } else {
                            let parsed_log = update_snapshot_from_progress_line(snapshot, &line);
                            if let Some(entry) = parsed_log {
                                snapshot.logs = push_log_line(&snapshot.logs, &entry, 10);
                            } else {
                                snapshot.logs = push_log_line(&snapshot.logs, &line, 10);
                            }
                        }
                    }
                }
            }

            let cancel_requested = runtime
                .lock()
                .ok()
                .map(|runtime_guard| runtime_guard.cancel_requested)
                .unwrap_or(false);
            if cancel_requested {
                let _ = child.kill();
            }

            match child.try_wait() {
                Ok(Some(status)) => {
                    while let Ok((is_stderr, line)) = line_receiver.try_recv() {
                        if let Ok(mut runtime_guard) = runtime.lock() {
                            if let Some(snapshot) = runtime_guard.snapshot.as_mut() {
                                if snapshot.job_id != job_id_for_thread {
                                    continue;
                                }

                                if is_stderr {
                                    let prefixed = format!("stderr: {line}");
                                    snapshot.logs = push_log_line(&snapshot.logs, &prefixed, 10);
                                } else {
                                    let parsed_log =
                                        update_snapshot_from_progress_line(snapshot, &line);
                                    if let Some(entry) = parsed_log {
                                        snapshot.logs = push_log_line(&snapshot.logs, &entry, 10);
                                    } else {
                                        snapshot.logs = push_log_line(&snapshot.logs, &line, 10);
                                    }
                                }
                            }
                        }
                    }

                    let files = collect_supported_files(&target_dir_for_thread);
                    let count = files.len();

                    if let Ok(mut runtime_guard) = runtime.lock() {
                        let cancelled = runtime_guard.cancel_requested;
                        runtime_guard.process_pid = None;
                        runtime_guard.cancel_requested = false;

                        if let Some(s) = runtime_guard.snapshot.as_mut() {
                            if s.job_id == job_id_for_thread {
                                s.finished_at = Some(now_timestamp_string());
                                for file_path in &files {
                                    let label =
                                        format_export_log_label(&target_dir_for_thread, file_path);
                                    if seen_files.insert(label.clone()) {
                                        s.logs = push_log_line(&s.logs, &label, 10);
                                    }
                                }
                                s.current = s.current.max(count);
                                if s.total.is_none() {
                                    s.total = Some(s.current.max(count));
                                }
                                if let Some(total) = s.total {
                                    if s.current > total {
                                        s.total = Some(s.current);
                                    }
                                }

                                if cancelled {
                                    s.status = "cancelled".to_string();
                                    s.message = "Export cancelled.".to_string();
                                } else if status.success() {
                                    if let Some(total) = s.total {
                                        s.current = total;
                                    } else {
                                        s.total = Some(s.current.max(count));
                                    }
                                    s.status = "completed".to_string();
                                    let total = s.total.unwrap_or(s.current);
                                    s.message = format!(
                                        "Export complete. Exported {} of {} files.",
                                        total, total
                                    );
                                } else {
                                    s.status = "failed".to_string();
                                    s.message = "Export failed.".to_string();
                                    if s.error.is_none() {
                                        s.error =
                                            Some(format!("Script exited with status {status}"));
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                Ok(None) => {
                    if last_fallback_scan.elapsed() >= Duration::from_secs(1) {
                        let files = collect_supported_files(&target_dir_for_thread);
                        let count = files.len();
                        if let Ok(mut runtime_guard) = runtime.lock() {
                            if let Some(s) = runtime_guard.snapshot.as_mut() {
                                if s.job_id == job_id_for_thread {
                                    s.current = s.current.max(count);

                                    for file_path in &files {
                                        let label = format_export_log_label(
                                            &target_dir_for_thread,
                                            file_path,
                                        );
                                        if seen_files.insert(label.clone()) {
                                            s.logs = push_log_line(&s.logs, &label, 10);
                                        }
                                    }

                                    if let Some(total) = s.total {
                                        s.message = format!(
                                            "Pulling notes... {} of {} files",
                                            s.current, total
                                        );
                                    } else {
                                        s.message = format!("Pulling notes... {} files", s.current);
                                    }
                                }
                            }
                        }
                        last_fallback_scan = Instant::now();
                    }
                    thread::sleep(Duration::from_millis(220));
                }
                Err(err) => {
                    if let Ok(mut runtime_guard) = runtime.lock() {
                        runtime_guard.process_pid = None;
                        if let Some(s) = runtime_guard.snapshot.as_mut() {
                            if s.job_id == job_id_for_thread {
                                s.status = "failed".to_string();
                                s.message = "Failed to poll export script.".to_string();
                                s.error = Some(err.to_string());
                                s.finished_at = Some(now_timestamp_string());
                            }
                        }
                    }
                    break;
                }
            }
        }
    });

    Ok(snapshot)
}

#[tauri::command]
fn get_export_job(
    export_state: State<ExportJobState>,
) -> Result<Option<ExportJobSnapshot>, String> {
    let guard = export_state
        .runtime
        .lock()
        .map_err(|_| "Lock error".to_string())?;
    Ok(guard.snapshot.clone())
}

#[tauri::command]
fn cancel_export_job(
    export_state: State<ExportJobState>,
) -> Result<Option<ExportJobSnapshot>, String> {
    let mut guard = export_state
        .runtime
        .lock()
        .map_err(|_| "Lock error".to_string())?;

    guard.cancel_requested = true;
    if let Some(pid) = guard.process_pid {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status();
    }

    if let Some(snapshot) = guard.snapshot.as_mut() {
        if snapshot.status == "running" || snapshot.status == "queued" {
            snapshot.status = "cancelling".to_string();
            snapshot.message = "Cancellation requested...".to_string();
        }
    }

    Ok(guard.snapshot.clone())
}

#[tauri::command]
fn delete_export_subfolder(export_folder_path: String) -> Result<String, String> {
    let target = app_owned_export_subfolder(&export_folder_path)?;
    if target.exists() {
        fs::remove_dir_all(&target)
            .map_err(|err| format!("Failed to delete export subfolder {target:?}: {err}"))?;
    }
    Ok(target.to_string_lossy().to_string())
}

#[tauri::command]
fn pick_export_folder() -> Result<Option<String>, String> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("osascript")
            .args([
                "-e",
                "POSIX path of (choose folder with prompt \"Select Notes Export Folder\")",
            ])
            .output()
            .map_err(|err| format!("Failed to run folder picker: {err}"))?;

        if output.status.success() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if value.is_empty() {
                return Ok(None);
            }
            return Ok(Some(value));
        }

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.contains("-128") {
            return Ok(None);
        }

        return Err(format!("Folder picker failed: {}", stderr.trim()));
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("Folder picker command is only implemented for macOS".to_string())
    }
}

#[tauri::command]
fn validate_export_folder(path: String) -> ExportFolderStatus {
    let trimmed = path.trim().to_string();
    let folder = Path::new(&trimmed);
    let exists = folder.exists();
    let is_dir = folder.is_dir();
    let (supported_files, permission_denied) = if exists && is_dir {
        count_supported_files(folder)
    } else {
        (0, false)
    };

    let message = if trimmed.is_empty() {
        "No folder selected.".to_string()
    } else if !exists {
        "Selected folder does not exist.".to_string()
    } else if !is_dir {
        "Selected path is not a directory.".to_string()
    } else if permission_denied {
        "Permission denied while reading selected folder. Grant access in macOS System Settings and retry validation.".to_string()
    } else {
        format!("Folder is valid. Found {supported_files} supported note file(s).")
    };

    ExportFolderStatus {
        path: trimmed,
        exists,
        is_dir,
        supported_files,
        permission_denied,
        message,
    }
}

#[tauri::command]
fn ollama_status() -> OllamaStatus {
    let version_output = run_command_capture("ollama", &["--version"]);
    let installed = version_output.is_ok();
    let version = version_output.ok().and_then(|(_, output)| {
        output
            .lines()
            .find(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
    });

    let running = is_ollama_running();
    let models = if installed {
        match run_command_capture("ollama", &["list"]) {
            Ok((true, output)) => parse_ollama_list_models(&output),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let message = if !installed {
        "Ollama CLI was not found. Install Ollama to continue.".to_string()
    } else if !running {
        format!(
            "Ollama is installed but not responding at {OLLAMA_BASE_URL}. Start Ollama and retry."
        )
    } else {
        "Ollama is installed and running.".to_string()
    };

    OllamaStatus {
        installed,
        running,
        version,
        models,
        message,
    }
}

#[tauri::command]
fn open_ollama_download_page() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open")
            .arg("https://ollama.com/download/mac")
            .status()
            .map_err(|err| format!("Failed to open browser for Ollama download: {err}"))?;
        if status.success() {
            return Ok(());
        }
        return Err("Browser open command failed while opening Ollama download page".to_string());
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("This helper is only implemented for macOS builds".to_string())
    }
}

#[tauri::command]
fn start_ollama() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(status) = Command::new("open").args(["-a", "Ollama"]).status() {
            if status.success() {
                return Ok("Requested launch of Ollama.app".to_string());
            }
        }
    }

    let _ = run_command_capture("ollama", &["--version"])?;
    Command::new("ollama")
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| format!("Failed to start ollama serve: {err}"))?;
    Ok("Started `ollama serve` in the background".to_string())
}

#[tauri::command]
async fn ollama_pull_model(model: String) -> Result<PullModelResult, String> {
    let model = model.trim().to_string();
    if model.is_empty() {
        return Err("model must not be empty".to_string());
    }

    tauri::async_runtime::spawn_blocking(move || {
        let (success, output, timed_out) =
            run_command_capture_with_timeout("ollama", &["pull", &model], model_pull_timeout())
                .map_err(|err| format!("Failed to run `ollama pull {model}`: {err}"))?;
        let (error_type, retryable) = if success {
            ("none".to_string(), false)
        } else {
            classify_pull_failure(&output, timed_out)
        };

        Ok(PullModelResult {
            model,
            success,
            output,
            error_type,
            retryable,
            timed_out,
        })
    })
    .await
    .map_err(|err| format!("Model pull task failed: {err}"))?
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState {
            runtime: Mutex::new(BackendRuntime {
                child: None,
                port: DEFAULT_BACKEND_PORT,
            }),
        })
        .manage(ExportJobState {
            runtime: Arc::new(Mutex::new(ExportJobRuntime::default())),
        })
        .invoke_handler(tauri::generate_handler![
            backend_base_url,
            backend_ingest,
            load_app_config,
            save_app_config,
            setup_load_config,
            setup_save_config,
            setup_save_notes_export_dir,
            load_chat_threads,
            save_chat_threads,
            start_export_job,
            get_export_job,
            cancel_export_job,
            delete_export_subfolder,
            ollama_status,
            open_ollama_download_page,
            start_ollama,
            ollama_pull_model,
            pick_export_folder,
            validate_export_folder
        ])
        .setup(|app| {
            let state: State<BackendState> = app.state();
            let mut runtime_guard = state.runtime.lock().map_err(|_| "Lock error")?;
            if runtime_guard.child.is_none() {
                let port = select_backend_port()?;
                let app_handle = app.handle();
                let app_config = app_config_path(&app_handle)
                    .ok()
                    .and_then(|path| read_app_config(&path).ok())
                    .unwrap_or_default();
                let mut child = spawn_backend(port, Some(&app_config))?;
                if let Err(err) = wait_for_backend_health(port, &mut child) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(err.into());
                }
                runtime_guard.port = port;
                runtime_guard.child = Some(child);
            }
            Ok(())
        })
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event.event() {
                let state: State<BackendState> = event.window().state();
                if let Ok(mut runtime_guard) = state.runtime.lock() {
                    if let Some(mut child) = runtime_guard.child.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::{
        app_owned_export_subfolder, backend_env_from_config, build_export_script_args,
        classify_pull_failure, count_supported_files, is_supported_export_file,
        normalize_app_config, parse_backend_port, parse_ollama_list_models, requires_onboarding,
    };
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parse_backend_port_none_is_none() {
        assert_eq!(parse_backend_port(None).expect("parse none"), None);
    }

    #[test]
    fn parse_backend_port_valid_number() {
        assert_eq!(
            parse_backend_port(Some("8001".to_string())).expect("parse 8001"),
            Some(8001)
        );
    }

    #[test]
    fn parse_backend_port_zero_is_error() {
        let err = parse_backend_port(Some("0".to_string())).expect_err("expected error");
        assert!(err.contains("Invalid BACKEND_PORT value"));
    }

    #[test]
    fn parse_backend_port_invalid_is_error() {
        let err = parse_backend_port(Some("abc".to_string())).expect_err("expected error");
        assert!(err.contains("Invalid BACKEND_PORT value"));
    }

    #[test]
    fn parse_ollama_list_models_skips_header_and_empty_lines() {
        let raw = "NAME            ID              SIZE    MODIFIED\nnomic-embed-text abc123          274 MB  2 days ago\n\nneural-chat     def456          4.1 GB  2 days ago";
        let models = parse_ollama_list_models(raw);
        assert_eq!(
            models,
            vec!["nomic-embed-text".to_string(), "neural-chat".to_string()]
        );
    }

    #[test]
    fn parse_ollama_list_models_ignores_duplicates() {
        let raw =
            "NAME ID SIZE MODIFIED\nnomic-embed-text abc 1 B now\nnomic-embed-text xyz 1 B now";
        let models = parse_ollama_list_models(raw);
        assert_eq!(models, vec!["nomic-embed-text".to_string()]);
    }

    #[test]
    fn classify_pull_failure_timeout_is_retryable() {
        let (error_type, retryable) = classify_pull_failure("timed out", true);
        assert_eq!(error_type, "timeout".to_string());
        assert!(retryable);
    }

    #[test]
    fn classify_pull_failure_disk_full_is_not_retryable() {
        let (error_type, retryable) =
            classify_pull_failure("write failed: no space left on device", false);
        assert_eq!(error_type, "disk_full".to_string());
        assert!(!retryable);
    }

    #[test]
    fn classify_pull_failure_offline_is_retryable() {
        let (error_type, retryable) = classify_pull_failure("connection refused", false);
        assert_eq!(error_type, "offline".to_string());
        assert!(retryable);
    }

    #[test]
    fn supported_export_file_extensions_are_detected() {
        assert!(is_supported_export_file(Path::new("note.html")));
        assert!(is_supported_export_file(Path::new("note.md")));
        assert!(is_supported_export_file(Path::new("note.TXT")));
        assert!(!is_supported_export_file(Path::new("note.pdf")));
    }

    #[test]
    fn count_supported_files_returns_count_and_permission_flag() {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        dir.push(format!(
            "notes_rag_app_count_supported_files_{}_{}",
            std::process::id(),
            nanos
        ));

        fs::create_dir_all(&dir).expect("create test dir");
        fs::write(dir.join("a.html"), "alpha").expect("write html");
        fs::write(dir.join("b.txt"), "beta").expect("write txt");
        fs::write(dir.join("c.pdf"), "gamma").expect("write pdf");

        let (count, permission_denied) = count_supported_files(&dir);
        assert_eq!(count, 2);
        assert!(!permission_denied);

        fs::remove_dir_all(&dir).expect("cleanup test dir");
    }

    #[test]
    fn normalize_app_config_migrates_legacy_fields() {
        let legacy = json!({
            "wizard_completed": true,
            "notes_export_dir": "/tmp/notes",
            "embed_model": "embed-v1",
            "chat_model": "chat-v1"
        });

        let config = normalize_app_config(&legacy);
        assert!(config.has_completed_onboarding);
        assert_eq!(config.export_folder_path.as_deref(), Some("/tmp/notes"));
        assert_eq!(config.embed_model, "embed-v1");
        assert_eq!(config.chat_model, "chat-v1");
    }

    #[test]
    fn requires_onboarding_true_without_export_path() {
        let config = super::AppConfig {
            has_completed_onboarding: true,
            export_folder_path: None,
            last_synced_at: None,
            embed_model: "nomic-embed-text".to_string(),
            chat_model: "neural-chat".to_string(),
        };
        assert!(requires_onboarding(&config));
    }

    #[test]
    fn requires_onboarding_true_without_last_synced_at() {
        let config = super::AppConfig {
            has_completed_onboarding: true,
            export_folder_path: Some("/tmp/notes".to_string()),
            last_synced_at: None,
            embed_model: "nomic-embed-text".to_string(),
            chat_model: "neural-chat".to_string(),
        };
        assert!(requires_onboarding(&config));
    }

    #[test]
    fn app_owned_export_subfolder_appends_fixed_suffix() {
        let target = app_owned_export_subfolder("/tmp/base").expect("target path");
        assert_eq!(target.to_string_lossy(), "/tmp/base/NotesLensExport");
    }

    #[test]
    fn build_export_script_args_returns_script_and_target() {
        let args = build_export_script_args(
            Path::new("/tmp/export_notes.applescript"),
            Path::new("/tmp/NotesLensExport"),
            None,
        );
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "/tmp/export_notes.applescript");
        assert_eq!(args[1], "/tmp/NotesLensExport");
    }

    #[test]
    fn build_export_script_args_includes_limit_when_set() {
        let args = build_export_script_args(
            Path::new("/tmp/export_notes.applescript"),
            Path::new("/tmp/NotesLensExport"),
            Some(20),
        );
        assert_eq!(args.len(), 3);
        assert_eq!(args[2], "20");
    }

    #[test]
    fn backend_env_from_config_uses_models_and_owned_export_dir() {
        let config = super::AppConfig {
            has_completed_onboarding: true,
            export_folder_path: Some("/tmp/notes-root".to_string()),
            last_synced_at: Some("2026-01-01T00:00:00Z".to_string()),
            embed_model: "nomic-embed-text".to_string(),
            chat_model: "neural-chat".to_string(),
        };

        let envs = backend_env_from_config(Some(&config));
        assert!(envs
            .iter()
            .any(|(k, v)| k == "OLLAMA_EMBEDDING_MODEL" && v == "nomic-embed-text"));
        assert!(envs
            .iter()
            .any(|(k, v)| k == "OLLAMA_CHAT_MODEL" && v == "neural-chat"));
        assert!(envs.iter().any(|(k, v)| {
            k == "NOTES_EXPORT_DIR" && v.ends_with("/tmp/notes-root/NotesLensExport")
        }));
    }

    #[test]
    fn backend_env_from_config_skips_empty_values() {
        let config = super::AppConfig {
            has_completed_onboarding: false,
            export_folder_path: Some("".to_string()),
            last_synced_at: None,
            embed_model: " ".to_string(),
            chat_model: "".to_string(),
        };

        let envs = backend_env_from_config(Some(&config));
        assert!(envs.is_empty());
    }
}
