#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, State};

struct BackendState {
    child: Mutex<Option<Child>>,
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

fn spawn_backend() -> Result<Child, String> {
    let backend_cmd = std::env::var("BACKEND_CMD").unwrap_or_else(|_| "python".to_string());
    let backend_port = std::env::var("BACKEND_PORT").unwrap_or_else(|_| "8001".to_string());
    let mut args = vec![
        "-m".to_string(),
        "uvicorn".to_string(),
        "main:app".to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        backend_port,
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

fn main() {
    tauri::Builder::default()
        .manage(BackendState {
            child: Mutex::new(None),
        })
        .setup(|app| {
            let state: State<BackendState> = app.state();
            let mut child_guard = state.child.lock().map_err(|_| "Lock error")?;
            if child_guard.is_none() {
                let child = spawn_backend()?;
                *child_guard = Some(child);
            }
            Ok(())
        })
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event.event() {
                api.prevent_close();
                let state: State<BackendState> = event.window().state();
                if let Ok(mut child_guard) = state.child.lock() {
                    if let Some(mut child) = child_guard.take() {
                        let _ = child.kill();
                    }
                }
                let _ = event.window().close();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
