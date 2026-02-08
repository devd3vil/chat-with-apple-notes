#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, State};

const DEFAULT_BACKEND_PORT: u16 = 8001;
const PORT_SCAN_SIZE: u16 = 100;
const HEALTH_CHECK_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_MS: u64 = 250;
const BACKEND_INGEST_TIMEOUT_SECS: u64 = 7200;
const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const SETUP_CONFIG_FILE: &str = "setup.json";
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

fn find_backend_workdir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("BACKEND_WORKDIR") {
        return std::path::PathBuf::from(dir);
    }

    let current = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
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

fn spawn_backend(port: u16) -> Result<Child, String> {
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

    Command::new(backend_cmd)
        .args(args)
        .current_dir(workdir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| format!("Failed to start backend: {err}"))
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

fn setup_config_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let config_dir = app
        .path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Unable to resolve app config directory".to_string())?;
    fs::create_dir_all(&config_dir)
        .map_err(|err| format!("Failed to create config dir {config_dir:?}: {err}"))?;
    Ok(config_dir.join(SETUP_CONFIG_FILE))
}

fn read_setup_config(path: &Path) -> Result<SetupConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read setup config {path:?}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("Invalid setup config JSON: {err}"))
}

fn write_setup_config(path: &Path, config: &SetupConfig) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(config)
        .map_err(|err| format!("Failed to serialize setup config: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("Failed to write setup config {path:?}: {err}"))
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
fn setup_load_config(app: tauri::AppHandle) -> Result<SetupConfig, String> {
    let path = setup_config_path(&app)?;
    if !path.exists() {
        return Ok(SetupConfig::default());
    }
    read_setup_config(&path)
}

#[tauri::command]
fn setup_save_config(
    app: tauri::AppHandle,
    embed_model: String,
    chat_model: String,
    wizard_completed: bool,
) -> Result<SetupConfig, String> {
    let embed_model = embed_model.trim().to_string();
    let chat_model = chat_model.trim().to_string();
    if embed_model.is_empty() {
        return Err("embed_model must not be empty".to_string());
    }
    if chat_model.is_empty() {
        return Err("chat_model must not be empty".to_string());
    }

    let path = setup_config_path(&app)?;
    let existing = if path.exists() {
        read_setup_config(&path)?
    } else {
        SetupConfig::default()
    };

    let config = SetupConfig {
        embed_model,
        chat_model,
        notes_export_dir: existing.notes_export_dir,
        wizard_completed,
    };
    write_setup_config(&path, &config)?;
    Ok(config)
}

#[tauri::command]
fn setup_save_notes_export_dir(
    app: tauri::AppHandle,
    notes_export_dir: Option<String>,
) -> Result<SetupConfig, String> {
    let path = setup_config_path(&app)?;
    let mut config = if path.exists() {
        read_setup_config(&path)?
    } else {
        SetupConfig::default()
    };
    config.notes_export_dir = notes_export_dir
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    write_setup_config(&path, &config)?;
    Ok(config)
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
    } else if supported_files == 0 {
        "Folder has no supported note files (.html/.htm/.md/.markdown/.txt).".to_string()
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
        .invoke_handler(tauri::generate_handler![
            backend_base_url,
            backend_ingest,
            setup_load_config,
            setup_save_config,
            setup_save_notes_export_dir,
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
                let mut child = spawn_backend(port)?;
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
        classify_pull_failure, count_supported_files, is_supported_export_file, parse_backend_port,
        parse_ollama_list_models,
    };
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
}
