use crate::audio_toolkit::constants;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
struct WorkerRequest<'a> {
    id: u64,
    audio_path: &'a str,
    language: Option<&'a str>,
    context: &'a str,
}

#[derive(Deserialize)]
struct ReadyResponse {
    ready: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct WorkerResponse {
    ok: bool,
    text: Option<String>,
    error: Option<String>,
    elapsed_ms: Option<u64>,
}

pub struct QwenMlxEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_request_id: u64,
}

impl QwenMlxEngine {
    pub fn load(model_path: &Path, worker_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!(
                "Qwen3 MLX model directory not found: {}",
                model_path.display()
            );
        }
        if !worker_path.exists() {
            anyhow::bail!("Qwen3 MLX worker not found: {}", worker_path.display());
        }

        let python = find_python_with_qwen3_mlx(model_path)?;
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
                    "failed to start Qwen3 MLX worker with Python '{}'",
                    python.display()
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .context("Qwen3 MLX worker stdin was not available")?;
        let stdout = child
            .stdout
            .take()
            .context("Qwen3 MLX worker stdout was not available")?;
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
            .context("failed to read Qwen3 MLX worker startup response")?;
        if n == 0 {
            let status = engine.child.try_wait().ok().flatten();
            anyhow::bail!(
                "Qwen3 MLX worker exited before becoming ready: {:?}",
                status
            );
        }

        let ready: ReadyResponse = serde_json::from_str(&line)
            .with_context(|| format!("invalid Qwen3 MLX worker startup response: {line:?}"))?;
        if !ready.ready {
            anyhow::bail!(
                "Qwen3 MLX worker failed to load model: {}",
                ready.error.unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(engine)
    }

    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: Option<&str>,
        context: &str,
    ) -> Result<String> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let audio_path = write_temp_wav(audio, request_id)?;
        let audio_path_string = audio_path.to_string_lossy().to_string();
        let request = WorkerRequest {
            id: request_id,
            audio_path: &audio_path_string,
            language,
            context,
        };

        let result = (|| -> Result<String> {
            serde_json::to_writer(&mut self.stdin, &request)
                .context("failed to serialize Qwen3 MLX request")?;
            self.stdin
                .write_all(b"\n")
                .context("failed to write Qwen3 MLX request newline")?;
            self.stdin
                .flush()
                .context("failed to flush Qwen3 MLX request")?;

            let mut line = String::new();
            let n = self
                .stdout
                .read_line(&mut line)
                .context("failed to read Qwen3 MLX response")?;
            if n == 0 {
                let status = self.child.try_wait().ok().flatten();
                anyhow::bail!("Qwen3 MLX worker exited unexpectedly: {:?}", status);
            }

            let response: WorkerResponse = serde_json::from_str(&line)
                .with_context(|| format!("invalid Qwen3 MLX worker response: {line:?}"))?;
            if !response.ok {
                anyhow::bail!(
                    "Qwen3 MLX transcription failed: {}",
                    response
                        .error
                        .unwrap_or_else(|| "unknown error".to_string())
                );
            }
            if let Some(elapsed_ms) = response.elapsed_ms {
                log::debug!("Qwen3 MLX worker inference took {}ms", elapsed_ms);
            }
            Ok(response.text.unwrap_or_default())
        })();

        let _ = fs::remove_file(&audio_path);
        result
    }
}

impl Drop for QwenMlxEngine {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"{\"cmd\":\"shutdown\"}\n");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_temp_wav(audio: &[f32], request_id: u64) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("handy-qwen3-mlx");
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create temp audio directory: {}", dir.display()))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!(
        "qwen3-mlx-{}-{}-{}.wav",
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

fn find_python_with_qwen3_mlx(model_path: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("HANDY_QWEN3_MLX_PYTHON") {
        let candidate = PathBuf::from(path);
        if python_can_import_qwen3(&candidate) {
            return Ok(candidate);
        }
        anyhow::bail!(
            "HANDY_QWEN3_MLX_PYTHON is set, but it cannot import mlx_qwen3_asr: {}",
            candidate.display()
        );
    }

    for marker in python_hint_files(model_path) {
        if let Ok(path) = fs::read_to_string(&marker) {
            let candidate = PathBuf::from(path.trim());
            if candidate.exists() && python_can_import_qwen3(&candidate) {
                return Ok(candidate);
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join(".asr-bench/bin/python");
            if candidate.exists() && python_can_import_qwen3(&candidate) {
                return Ok(candidate);
            }
        }
    }

    for candidate in ["python3", "python"] {
        let path = PathBuf::from(candidate);
        if python_can_import_qwen3(&path) {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "No Python with mlx-qwen3-asr found. Install mlx-qwen3-asr or set HANDY_QWEN3_MLX_PYTHON."
    );
}

fn python_hint_files(model_path: &Path) -> Vec<PathBuf> {
    let mut files = vec![model_path.join(".handy-python")];
    if let Some(parent) = model_path.parent() {
        files.push(parent.join("qwen3_mlx_python.txt"));
    }
    files
}

fn python_can_import_qwen3(python: &Path) -> bool {
    Command::new(python)
        .arg("-c")
        .arg("import mlx_qwen3_asr")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
