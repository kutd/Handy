use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

#[derive(Clone, Debug)]
pub struct RecentPostProcessContext {
    pub raw_transcription: String,
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub inserted_char_count: usize,
    recorded_at: Instant,
}

pub struct PostProcessContextState {
    last: Mutex<Option<RecentPostProcessContext>>,
}

impl PostProcessContextState {
    pub fn new() -> Self {
        Self {
            last: Mutex::new(None),
        }
    }
}

pub fn record(
    app: &AppHandle,
    raw_transcription: String,
    final_text: String,
    post_processed_text: Option<String>,
    inserted_char_count: usize,
) {
    if final_text.trim().is_empty() {
        return;
    }

    let Some(state) = app.try_state::<PostProcessContextState>() else {
        return;
    };

    if let Ok(mut last) = state.last.lock() {
        *last = Some(RecentPostProcessContext {
            raw_transcription,
            final_text,
            post_processed_text,
            inserted_char_count,
            recorded_at: Instant::now(),
        });
    };
}

pub fn recent(app: &AppHandle, max_age: Duration) -> Option<RecentPostProcessContext> {
    let state = app.try_state::<PostProcessContextState>()?;
    let mut last = state.last.lock().ok()?;
    let value = last.clone()?;

    if value.recorded_at.elapsed() > max_age {
        *last = None;
        return None;
    }

    Some(value)
}

pub fn clear(app: &AppHandle) {
    let Some(state) = app.try_state::<PostProcessContextState>() else {
        return;
    };

    if let Ok(mut last) = state.last.lock() {
        *last = None;
    };
}
