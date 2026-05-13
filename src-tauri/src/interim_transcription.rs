use crate::audio_toolkit::InterimTranscriptionAudio;
use crate::managers::audio::AudioRecordingManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::get_settings;
use log::{debug, error};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, Serialize)]
struct InterimTranscriptionEvent {
    text: String,
    sample_count: usize,
    sample_start: usize,
    sample_end: usize,
    replace_existing: bool,
}

#[derive(Clone)]
pub struct InterimTranscriptionState {
    generation: Arc<AtomicU64>,
    active: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    pending_settled: Arc<Mutex<Option<(u64, InterimTranscriptionAudio)>>>,
}

impl InterimTranscriptionState {
    pub fn new() -> Self {
        Self {
            generation: Arc::new(AtomicU64::new(0)),
            active: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            pending_settled: Arc::new(Mutex::new(None)),
        }
    }
}

pub fn start_session(app: &AppHandle) {
    if let Some(state) = app.try_state::<InterimTranscriptionState>() {
        state.generation.fetch_add(1, Ordering::SeqCst);
        state.active.store(true, Ordering::SeqCst);
        state.running.store(false, Ordering::SeqCst);
        if let Ok(mut pending) = state.pending_settled.lock() {
            *pending = None;
        }
    }
}

pub fn finish_session(app: &AppHandle) {
    if let Some(state) = app.try_state::<InterimTranscriptionState>() {
        state.generation.fetch_add(1, Ordering::SeqCst);
        state.active.store(false, Ordering::SeqCst);
        state.running.store(false, Ordering::SeqCst);
        if let Ok(mut pending) = state.pending_settled.lock() {
            *pending = None;
        }
    }
}

pub fn request_preview(app: &AppHandle, audio: InterimTranscriptionAudio) {
    let settings = get_settings(app);
    if !settings.experimental_enabled
        || settings.overlay_position == crate::settings::OverlayPosition::None
    {
        return;
    }

    let Some(state) = app.try_state::<InterimTranscriptionState>() else {
        return;
    };

    if !state.active.load(Ordering::SeqCst) {
        return;
    }

    if state.running.swap(true, Ordering::SeqCst) {
        if audio.replace_existing {
            let generation = state.generation.load(Ordering::SeqCst);
            if let Ok(mut pending) = state.pending_settled.lock() {
                *pending = Some((generation, audio));
            }
        }
        debug!("Skipping interim transcription preview because one is already running");
        return;
    }

    let generation = state.generation.load(Ordering::SeqCst);
    let generation_ref = state.generation.clone();
    let active_ref = state.active.clone();
    let running_ref = state.running.clone();
    let pending_ref = state.pending_settled.clone();
    let app_for_task = app.clone();
    let sample_count = audio.samples.len();
    let sample_start = audio.sample_start;
    let sample_end = audio.sample_end;
    let replace_existing = audio.replace_existing;

    std::thread::spawn(move || {
        let maybe_emit = || {
            let tm = app_for_task.try_state::<Arc<TranscriptionManager>>()?;

            let result = tm.try_transcribe_preview(audio.samples);
            let text = match result {
                Ok(Some(text)) => text.trim().to_string(),
                Ok(None) => return None,
                Err(err) => {
                    debug!("Interim transcription preview failed: {}", err);
                    return None;
                }
            };

            if text.is_empty()
                || !active_ref.load(Ordering::SeqCst)
                || generation_ref.load(Ordering::SeqCst) != generation
            {
                return None;
            }

            let is_recording = app_for_task
                .try_state::<Arc<AudioRecordingManager>>()
                .map_or(false, |audio| audio.is_recording());
            if !is_recording {
                return None;
            }

            crate::overlay::set_recording_overlay_expanded(&app_for_task, true);

            if let Err(err) = app_for_task.emit(
                "interim-transcription",
                InterimTranscriptionEvent {
                    sample_count,
                    sample_start,
                    sample_end,
                    text,
                    replace_existing,
                },
            ) {
                error!("Failed to emit interim transcription preview: {}", err);
            }

            Some(())
        };

        let _ = maybe_emit();
        running_ref.store(false, Ordering::SeqCst);

        let pending = pending_ref
            .lock()
            .ok()
            .and_then(|mut pending| pending.take());
        if let Some((pending_generation, pending_audio)) = pending {
            if active_ref.load(Ordering::SeqCst)
                && generation_ref.load(Ordering::SeqCst) == pending_generation
            {
                request_preview(&app_for_task, pending_audio);
            }
        }
    });
}
