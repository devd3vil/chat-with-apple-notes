#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, State};

const DEFAULT_BACKEND_PORT: u16 = 8001;
const PORT_SCAN_SIZE: u16 = 100;
const HEALTH_CHECK_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_MS: u64 = 250;

struct BackendRuntime {
    child: Option<Child>,
    port: u16,
}

struct BackendState {
    runtime: Mutex<BackendRuntime>,
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

#[tauri::command]
fn backend_base_url(state: State<BackendState>) -> Result<String, String> {
    let runtime_guard = state.runtime.lock().map_err(|_| "Lock error".to_string())?;
    Ok(format!("http://127.0.0.1:{}", runtime_guard.port))
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState {
            runtime: Mutex::new(BackendRuntime {
                child: None,
                port: DEFAULT_BACKEND_PORT,
            }),
        })
        .invoke_handler(tauri::generate_handler![backend_base_url])
        .setup(|app| {
            let state: State<BackendState> = app.state();
            let mut runtime_guard = state.runtime.lock().map_err(|_| "Lock error")?;
            if runtime_guard.child.is_none() {
                let port = select_backend_port()?;
                let mut child = spawn_backend(port)?;
                wait_for_backend_health(port, &mut child)?;
                runtime_guard.port = port;
                runtime_guard.child = Some(child);
            }
            Ok(())
        })
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event.event() {
                api.prevent_close();
                let state: State<BackendState> = event.window().state();
                if let Ok(mut runtime_guard) = state.runtime.lock() {
                    if let Some(mut child) = runtime_guard.child.take() {
                        let _ = child.kill();
                    }
                }
                let _ = event.window().close();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::parse_backend_port;

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
}
