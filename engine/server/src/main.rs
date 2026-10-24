//! `audio-server` (Rust engine): the port of `server/main.py`. Same CLI,
//! config files, WS/OSC/HTTP contracts; see `docs/reference/rust_server.md`.

mod app;
mod audio;
mod config;
mod dispatcher;
mod http;
mod osc;
mod persist;
mod priority;
mod py;
mod ring;
mod shared;
mod validate;
mod workers;
mod ws;

use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

// Debug builds abort on any allocation inside `assert_no_alloc` (the audio
// callback and the worker hot paths). A no-op allocator wrapper in release.
#[cfg(debug_assertions)]
#[global_allocator]
static ALLOC: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

#[derive(Parser, Debug)]
#[command(name = "audio-server", about = "Realtime audio feature server")]
struct Cli {
    /// path to main config file (default: <repo>/configs/main.yaml)
    #[arg(long)]
    config: Option<String>,
    /// disable the WebSocket server (headless OSC)
    #[arg(long)]
    no_ws: bool,
    /// open the bundled UI in the default browser
    #[arg(long)]
    open: bool,
    /// override input device index
    #[arg(long, allow_negative_numbers = true)]
    device: Option<i64>,
    /// logging level
    #[arg(long, default_value = "INFO")]
    log_level: String,
}

/// The repo checkout this binary belongs to: the first ancestor of the
/// executable holding `configs/main.example.yaml`, else the workspace this
/// crate was built from, else the cwd.
fn repo_root() -> PathBuf {
    let is_repo = |p: &Path| p.join("configs").join("main.example.yaml").exists();
    if let Ok(exe) = std::env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        for anc in exe.ancestors().skip(1) {
            if is_repo(anc) {
                return anc.to_path_buf();
            }
        }
    }
    let built = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if is_repo(&built) {
        return built.canonicalize().unwrap_or(built);
    }
    std::env::current_dir().unwrap_or_default()
}

fn absolute(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    }
}

/// `_resolve_config_path`: default `<repo>/configs/main.yaml`; absolute as
/// given; relative → cwd first, then the repo root. Resolved like Python's
/// `Path.resolve()`, so a symlinked main.yaml is written through (the atomic
/// rename would otherwise replace the link with a plain file) and presets
/// land next to the real file.
fn resolve_config_path(arg: Option<&str>, repo: &Path) -> PathBuf {
    let canonical = repo.join("configs").join("main.yaml");
    let Some(a) = arg else { return resolve(&canonical) };
    let p = Path::new(a);
    if p.is_absolute() {
        return resolve(p);
    }
    let cwd_c = resolve(&absolute(p));
    if cwd_c.exists() {
        return cwd_c;
    }
    let repo_c = resolve(&repo.join(p));
    if repo_c.exists() {
        return repo_c;
    }
    cwd_c
}

/// `Path.resolve()` (non-strict): symlinks followed; a missing file resolves
/// through its parent directory.
fn resolve(p: &Path) -> PathBuf {
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    match (p.parent().and_then(|d| d.canonicalize().ok()), p.file_name()) {
        (Some(dir), Some(name)) => dir.join(name),
        _ => p.to_path_buf(),
    }
}

fn fatal(message: &str, hint: &str) -> i32 {
    eprintln!("audio-server: error: {message}");
    if !hint.is_empty() {
        eprintln!("audio-server: hint: {hint}");
    }
    1
}

fn init_logging(level: &str) {
    let lvl = match level.to_ascii_uppercase().as_str() {
        "DEBUG" => tracing::Level::DEBUG,
        "WARNING" | "WARN" => tracing::Level::WARN,
        "ERROR" | "CRITICAL" => tracing::Level::ERROR,
        "TRACE" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };
    tracing_subscriber::fmt()
        .with_max_level(lvl)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
}

fn main() {
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    let cli = Cli::parse();
    init_logging(&cli.log_level);
    priority::boost_current_thread("main");
    let repo = repo_root();
    let config_path = resolve_config_path(cli.config.as_deref(), &repo);
    if !config_path.exists() && cli.config.is_none() {
        // Fresh clone: seed configs/main.yaml from the checked-in example.
        let example = config_path.with_file_name("main.example.yaml");
        if example.exists() {
            match std::fs::copy(&example, &config_path) {
                Ok(_) => tracing::info!("seeded {} from {}", config_path.display(), example.display()),
                Err(e) => tracing::warn!("could not seed {}: {e}", config_path.display()),
            }
        }
    }
    if !config_path.exists() {
        return fatal(
            &format!("config file not found: {}", config_path.display()),
            "omit --config to use configs/main.yaml (seeded from main.example.yaml)",
        );
    }
    tracing::info!("loading config from {}", config_path.display());
    let mut example = config_path.with_file_name("main.example.yaml");
    if !example.exists() {
        example = repo.join("configs").join("main.example.yaml");
    }
    let mut cfg = match config::load_config_or_fallback(&config_path, &example) {
        Ok(c) => c,
        Err(e) => {
            let name = config_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            return fatal(
                &format!("invalid config: {e}"),
                &format!(
                    "fix the value in {name}, or delete the file to regenerate it from configs/main.example.yaml"
                ),
            );
        }
    };
    if cli.no_ws {
        // As the reference: the flag lands on the config object (and so in
        // anything persisted during this run).
        cfg.ws.enabled = false;
    }
    let args = app::Args { no_ws: cli.no_ws, open: cli.open, device: cli.device };

    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => return fatal(&format!("could not start the async runtime: {e}"), ""),
    };
    let ui_root = repo.join("ui");
    rt.block_on(async move {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = match app::App::start(args, cfg, config_path, ui_root, tx.clone()) {
            Ok(a) => a,
            Err(e) => return fatal(&e.message, &e.hint),
        };
        let st = ws::WsState { tx, next_id: Arc::new(AtomicU64::new(1)) };
        if let Err(e) = app.spawn_servers(st) {
            return fatal(&e.message, &e.hint);
        }
        app.run(rx).await;
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_resolution() {
        let repo = repo_root();
        assert!(repo.join("configs/main.example.yaml").exists());
        assert_eq!(resolve_config_path(None, &repo), repo.join("configs/main.yaml"));
        assert_eq!(resolve_config_path(Some("/abs/x.yaml"), &repo), PathBuf::from("/abs/x.yaml"));
        // Relative: cwd miss → repo hit.
        let got = resolve_config_path(Some("configs/main.example.yaml"), &repo);
        assert!(got.exists());
    }

    /// A symlinked main.yaml resolves to its target, so the atomic persist
    /// writes through instead of replacing the link.
    #[cfg(unix)]
    #[test]
    fn config_path_follows_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.yaml");
        std::fs::write(&real, "x: 1\n").unwrap();
        let link = dir.path().join("main.yaml");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let got = resolve_config_path(Some(link.to_str().unwrap()), Path::new("/nonexistent"));
        assert_eq!(got, real.canonicalize().unwrap());
    }
}
