use crate::audio_toolkit::constants;
use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
struct ReadyResponse {
    ready: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct WorkerResponse {
    ok: bool,
    text: Option<String>,
    stable_text: Option<String>,
    error: Option<String>,
    elapsed_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SherpaStreamUpdate {
    pub text: String,
    pub stable_text: String,
}

pub struct SherpaOnnxEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_request_id: u64,
}

impl SherpaOnnxEngine {
    pub fn load(model_path: &Path, worker_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!(
                "sherpa-onnx model directory not found: {}",
                model_path.display()
            );
        }
        if !worker_path.exists() {
            anyhow::bail!("sherpa-onnx worker not found: {}", worker_path.display());
        }

        let python = find_python_with_sherpa_onnx(model_path, worker_path)?;
        let mut child = Command::new(&python)
            .arg(worker_path)
            .arg(model_path)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start sherpa-onnx worker with Python '{}'",
                    python.display()
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .context("sherpa-onnx worker stdin was not available")?;
        let stdout = child
            .stdout
            .take()
            .context("sherpa-onnx worker stdout was not available")?;
        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request_id: 1,
        };

        let mut line = String::new();
        let n = engine
            .stdout
            .read_line(&mut line)
            .context("failed to read sherpa-onnx worker startup response")?;
        if n == 0 {
            let status = engine.child.try_wait().ok().flatten();
            anyhow::bail!(
                "sherpa-onnx worker exited before becoming ready: {:?}",
                status
            );
        }

        let ready: ReadyResponse = serde_json::from_str(&line)
            .with_context(|| format!("invalid sherpa-onnx worker startup response: {line:?}"))?;
        if !ready.ready {
            anyhow::bail!(
                "sherpa-onnx worker failed to load model: {}",
                ready.error.unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(engine)
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let audio_path = write_temp_wav(audio, request_id)?;
        let audio_path_string = audio_path.to_string_lossy().to_string();
        let request = serde_json::json!({
            "cmd": "transcribe",
            "id": request_id,
            "audio_path": audio_path_string,
        });

        let result = (|| -> Result<String> {
            let response = self.send_json_request(&request)?;
            if !response.ok {
                anyhow::bail!(
                    "sherpa-onnx transcription failed: {}",
                    response
                        .error
                        .unwrap_or_else(|| "unknown error".to_string())
                );
            }
            if let Some(elapsed_ms) = response.elapsed_ms {
                log::debug!("sherpa-onnx worker inference took {}ms", elapsed_ms);
            }
            Ok(response.text.unwrap_or_default())
        })();

        let _ = fs::remove_file(&audio_path);
        result
    }

    pub fn stream_start(&mut self, hotwords: &str) -> Result<()> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let request = serde_json::json!({
            "cmd": "stream_start",
            "id": request_id,
            "hotwords": hotwords,
        });

        let response = self.send_json_request(&request)?;
        if !response.ok {
            anyhow::bail!(
                "sherpa-onnx streaming start failed: {}",
                response
                    .error
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }
        Ok(())
    }

    pub fn stream_feed(&mut self, audio: &[f32]) -> Result<SherpaStreamUpdate> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let mut pcm16 = Vec::with_capacity(audio.len() * 2);
        for sample in audio {
            let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            pcm16.extend_from_slice(&scaled.to_le_bytes());
        }

        let request = serde_json::json!({
            "cmd": "stream_feed",
            "id": request_id,
            "pcm16_b64": general_purpose::STANDARD.encode(pcm16),
        });

        let response = self.send_json_request(&request)?;
        if !response.ok {
            anyhow::bail!(
                "sherpa-onnx streaming feed failed: {}",
                response
                    .error
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }
        if let Some(elapsed_ms) = response.elapsed_ms {
            log::debug!("sherpa-onnx streaming feed took {}ms", elapsed_ms);
        }
        Ok(SherpaStreamUpdate {
            text: response.text.unwrap_or_default(),
            stable_text: response.stable_text.unwrap_or_default(),
        })
    }

    pub fn stream_cancel(&mut self) -> Result<()> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let request = serde_json::json!({
            "cmd": "stream_cancel",
            "id": request_id,
        });

        let response = self.send_json_request(&request)?;
        if !response.ok {
            anyhow::bail!(
                "sherpa-onnx streaming cancel failed: {}",
                response
                    .error
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }
        Ok(())
    }

    fn send_json_request(&mut self, request: &serde_json::Value) -> Result<WorkerResponse> {
        serde_json::to_writer(&mut self.stdin, request)
            .context("failed to serialize sherpa-onnx request")?;
        self.stdin
            .write_all(b"\n")
            .context("failed to write sherpa-onnx request newline")?;
        self.stdin
            .flush()
            .context("failed to flush sherpa-onnx request")?;

        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .context("failed to read sherpa-onnx response")?;
        if n == 0 {
            let status = self.child.try_wait().ok().flatten();
            anyhow::bail!("sherpa-onnx worker exited unexpectedly: {:?}", status);
        }

        serde_json::from_str(&line)
            .with_context(|| format!("invalid sherpa-onnx worker response: {line:?}"))
    }
}

impl Drop for SherpaOnnxEngine {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"{\"cmd\":\"shutdown\"}\n");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_temp_wav(audio: &[f32], request_id: u64) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("handy-sherpa-onnx");
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create temp audio directory: {}", dir.display()))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!(
        "sherpa-onnx-{}-{}-{}.wav",
        std::process::id(),
        timestamp,
        request_id
    ));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: constants::WHISPER_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)
        .with_context(|| format!("failed to create temp wav: {}", path.display()))?;
    for sample in audio {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(scaled)
            .with_context(|| format!("failed to write temp wav: {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize temp wav: {}", path.display()))?;
    Ok(path)
}

fn find_python_with_sherpa_onnx(model_path: &Path, worker_path: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("HANDY_SHERPA_ONNX_PYTHON") {
        let candidate = PathBuf::from(path);
        if python_can_import_sherpa_onnx(&candidate) {
            return Ok(candidate);
        }
        anyhow::bail!(
            "HANDY_SHERPA_ONNX_PYTHON is set, but it cannot import sherpa_onnx: {}",
            candidate.display()
        );
    }

    for marker in python_hint_files(model_path) {
        if let Ok(path) = fs::read_to_string(&marker) {
            let candidate = PathBuf::from(path.trim());
            if candidate.exists() && python_can_import_sherpa_onnx(&candidate) {
                return Ok(candidate);
            }
        }
    }

    if let Ok(managed_python) = managed_python_path(model_path) {
        if managed_python.exists() && python_can_import_sherpa_onnx(&managed_python) {
            return Ok(managed_python);
        }
    }

    for candidate in ["python3", "python"] {
        let path = PathBuf::from(candidate);
        if python_can_import_sherpa_onnx(&path) {
            return Ok(path);
        }
    }

    match ensure_managed_python_with_sherpa_onnx(model_path, worker_path) {
        Ok(python) => return Ok(python),
        Err(err) => {
            log::warn!("sherpa-onnx managed Python setup failed: {}", err);
        }
    }

    anyhow::bail!(
        "No Python with sherpa-onnx found. Handy tried to create a private runtime automatically, but setup failed. Check your internet connection or set HANDY_SHERPA_ONNX_PYTHON."
    );
}

fn python_hint_files(model_path: &Path) -> Vec<PathBuf> {
    let mut files = vec![model_path.join(".handy-sherpa-python")];
    if let Some(parent) = model_path.parent() {
        files.push(parent.join("sherpa_onnx_python.txt"));
    }
    files
}

fn python_can_import_sherpa_onnx(python: &Path) -> bool {
    Command::new(python)
        .arg("-c")
        .arg("import sherpa_onnx")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn ensure_managed_python_with_sherpa_onnx(
    model_path: &Path,
    worker_path: &Path,
) -> Result<PathBuf> {
    let runtime_dir = managed_runtime_dir(model_path)?;
    let python = managed_python_path(model_path)?;

    fs::create_dir_all(&runtime_dir).with_context(|| {
        format!(
            "failed to create sherpa-onnx runtime directory: {}",
            runtime_dir.display()
        )
    })?;

    if !python.exists() {
        if let Some(base_python) = find_bootstrap_python() {
            log::info!(
                "Creating sherpa-onnx Python runtime with {}",
                base_python.display()
            );
            let mut command = Command::new(&base_python);
            command.arg("-m").arg("venv").arg(&runtime_dir);
            run_and_check(&mut command, "create sherpa-onnx Python venv")?;
        } else {
            let uv = find_uv(worker_path)?;
            log::info!(
                "Creating sherpa-onnx Python runtime with uv at {}",
                uv.display()
            );
            let mut command = Command::new(&uv);
            command
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&runtime_dir)
                .env("UV_NO_PROGRESS", "1");
            run_and_check(&mut command, "create sherpa-onnx Python venv with uv")?;
        }
    }

    install_sherpa_onnx_package(&python, worker_path)?;
    if python_can_import_sherpa_onnx(&python) {
        return Ok(python);
    }

    anyhow::bail!(
        "managed Python runtime was created, but it cannot import sherpa_onnx: {}",
        python.display()
    );
}

fn install_sherpa_onnx_package(python: &Path, worker_path: &Path) -> Result<()> {
    if python_can_import_sherpa_onnx(python) {
        return Ok(());
    }

    let package = "sherpa-onnx==1.13.1";
    if let Ok(uv) = find_uv(worker_path) {
        log::info!("Installing {} into sherpa-onnx runtime with uv", package);
        let mut command = Command::new(&uv);
        command
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(python)
            .arg(package)
            .env("UV_NO_PROGRESS", "1");
        run_and_check(&mut command, "install sherpa-onnx with uv")?;
        return Ok(());
    }

    log::info!("Installing {} into sherpa-onnx runtime with pip", package);
    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg(package)
        .env("PIP_DISABLE_PIP_VERSION_CHECK", "1");
    run_and_check(&mut command, "install sherpa-onnx with pip")?;
    Ok(())
}

fn managed_runtime_dir(model_path: &Path) -> Result<PathBuf> {
    let models_dir = model_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model path has no parent: {}", model_path.display()))?;
    Ok(models_dir.join(".sherpa-onnx-runtime"))
}

fn managed_python_path(model_path: &Path) -> Result<PathBuf> {
    let runtime_dir = managed_runtime_dir(model_path)?;
    if cfg!(target_os = "windows") {
        Ok(runtime_dir.join("Scripts").join("python.exe"))
    } else {
        Ok(runtime_dir.join("bin").join("python"))
    }
}

fn find_bootstrap_python() -> Option<PathBuf> {
    let candidates = [
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
    ];

    candidates
        .iter()
        .map(PathBuf::from)
        .find(|candidate| python_version_can_run_sherpa(candidate))
}

fn python_version_can_run_sherpa(python: &Path) -> bool {
    Command::new(python)
        .arg("-c")
        .arg("import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn find_uv(worker_path: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("HANDY_SHERPA_ONNX_UV") {
        let candidate = PathBuf::from(path);
        if uv_works(&candidate) {
            return Ok(candidate);
        }
    }

    if let Some(resource_dir) = worker_path.parent() {
        for name in ["uv-aarch64-apple-darwin", "uv"] {
            let candidate = resource_dir.join(name);
            if candidate.exists() && uv_works(&candidate) {
                return Ok(candidate);
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join("src-tauri/resources/uv-aarch64-apple-darwin");
            if candidate.exists() && uv_works(&candidate) {
                return Ok(candidate);
            }
        }
    }

    let candidate = PathBuf::from("uv");
    if uv_works(&candidate) {
        return Ok(candidate);
    }

    anyhow::bail!("uv executable not found for sherpa-onnx Python setup")
}

fn uv_works(uv: &Path) -> bool {
    Command::new(uv)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_and_check(command: &mut Command, description: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to run command for {description}"))?;
    if output.status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!(
        "{} failed with status {}.\nstdout:\n{}\nstderr:\n{}",
        description,
        output.status,
        stdout.trim(),
        stderr.trim()
    );
}
