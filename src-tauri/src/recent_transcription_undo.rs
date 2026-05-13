use log::{debug, error, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

const DELETE_RECENT_TRANSCRIPTION_BINDING_ID: &str = "delete_recent_transcription";
const CLEAR_CURRENT_INPUT_WINDOW: Duration = Duration::from_millis(1500);

#[derive(Clone, Copy)]
struct UndoCandidate {
    char_count: usize,
    expires_at: Instant,
}

pub struct RecentTranscriptionUndo {
    candidate: Mutex<Option<UndoCandidate>>,
    clear_current_input_expires_at: Mutex<Option<Instant>>,
    listener_started: AtomicBool,
}

impl RecentTranscriptionUndo {
    pub fn new() -> Self {
        Self {
            candidate: Mutex::new(None),
            clear_current_input_expires_at: Mutex::new(None),
            listener_started: AtomicBool::new(false),
        }
    }
}

pub fn record_inserted_text(app: &AppHandle, text: &str) {
    let char_count = text.chars().count();
    if char_count == 0 {
        clear_candidate(app);
        return;
    }

    let settings = crate::settings::get_settings(app);
    if !settings.recent_transcription_undo_enabled {
        clear_candidate(app);
        return;
    }
    let undo_window = Duration::from_millis(settings.recent_transcription_undo_window_ms);

    let Some(state) = app.try_state::<RecentTranscriptionUndo>() else {
        warn!("RecentTranscriptionUndo state is not initialized");
        return;
    };

    clear_pending_current_input_clear(&state);

    match state.candidate.lock() {
        Ok(mut candidate) => {
            *candidate = Some(UndoCandidate {
                char_count,
                expires_at: Instant::now() + undo_window,
            });
            debug!(
                "Recorded recent transcription undo candidate: {} chars",
                char_count
            );
        }
        Err(err) => warn!("Failed to lock recent transcription undo state: {}", err),
    };
}

pub fn clear_candidate(app: &AppHandle) {
    if let Some(state) = app.try_state::<RecentTranscriptionUndo>() {
        if let Ok(mut candidate) = state.candidate.lock() {
            *candidate = None;
        }
        clear_pending_current_input_clear(&state);
    }
}

fn clear_pending_current_input_clear(state: &RecentTranscriptionUndo) {
    if let Ok(mut expires_at) = state.clear_current_input_expires_at.lock() {
        *expires_at = None;
    }
}

fn take_valid_candidate(app: &AppHandle) -> Option<usize> {
    let settings = crate::settings::get_settings(app);
    if !settings.recent_transcription_undo_enabled {
        clear_candidate(app);
        return None;
    }

    let state = app.try_state::<RecentTranscriptionUndo>()?;
    let mut candidate = state.candidate.lock().ok()?;
    let candidate_value = candidate.take()?;

    if Instant::now() > candidate_value.expires_at {
        debug!("Recent transcription undo candidate expired");
        return None;
    }

    Some(candidate_value.char_count)
}

fn arm_current_input_clear(app: &AppHandle) {
    let Some(state) = app.try_state::<RecentTranscriptionUndo>() else {
        return;
    };

    match state.clear_current_input_expires_at.lock() {
        Ok(mut expires_at) => {
            *expires_at = Some(Instant::now() + CLEAR_CURRENT_INPUT_WINDOW);
            debug!("Armed current input clear shortcut follow-up");
        }
        Err(err) => warn!("Failed to lock current input clear state: {}", err),
    };
}

fn take_valid_current_input_clear(app: &AppHandle) -> bool {
    let settings = crate::settings::get_settings(app);
    if !settings.recent_transcription_undo_enabled {
        clear_candidate(app);
        return false;
    }

    let Some(state) = app.try_state::<RecentTranscriptionUndo>() else {
        return false;
    };

    let Ok(mut expires_at) = state.clear_current_input_expires_at.lock() else {
        return false;
    };

    let Some(expires_at_value) = expires_at.take() else {
        return false;
    };

    if Instant::now() > expires_at_value {
        debug!("Current input clear shortcut follow-up expired");
        return false;
    }

    true
}

fn has_pending_delete_action(app: &AppHandle) -> bool {
    let settings = crate::settings::get_settings(app);
    if !settings.recent_transcription_undo_enabled {
        clear_candidate(app);
        return false;
    }

    true
}

#[cfg(target_os = "macos")]
fn send_backspaces(app: &AppHandle, char_count: usize) -> Result<(), String> {
    use crate::input::EnigoState;
    use enigo::{Direction, Key, Keyboard};

    let enigo_state = app
        .try_state::<EnigoState>()
        .ok_or("Input system is not initialized")?;
    let mut enigo = enigo_state
        .0
        .lock()
        .map_err(|err| format!("Failed to lock input system: {}", err))?;

    for index in 0..char_count {
        enigo
            .key(Key::Backspace, Direction::Click)
            .map_err(|err| format!("Failed to send Backspace: {}", err))?;
        if index % 32 == 31 {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn send_clear_current_input(app: &AppHandle) -> Result<(), String> {
    use crate::input::EnigoState;
    use enigo::{Direction, Key, Keyboard};

    let enigo_state = app
        .try_state::<EnigoState>()
        .ok_or("Input system is not initialized")?;
    let mut enigo = enigo_state
        .0
        .lock()
        .map_err(|err| format!("Failed to lock input system: {}", err))?;

    enigo
        .key(Key::Meta, Direction::Press)
        .map_err(|err| format!("Failed to press Command: {}", err))?;
    enigo
        .key(Key::Other(0), Direction::Click)
        .map_err(|err| format!("Failed to send Command+A: {}", err))?;
    std::thread::sleep(Duration::from_millis(20));
    enigo
        .key(Key::Meta, Direction::Release)
        .map_err(|err| format!("Failed to release Command: {}", err))?;
    std::thread::sleep(Duration::from_millis(20));
    enigo
        .key(Key::Backspace, Direction::Click)
        .map_err(|err| format!("Failed to send Backspace: {}", err))?;

    Ok(())
}

#[cfg(target_os = "macos")]
fn clear_current_input(app: &AppHandle) {
    let app_for_main = app.clone();
    if let Err(err) =
        app.run_on_main_thread(move || match send_clear_current_input(&app_for_main) {
            Ok(()) => {
                crate::post_process_context::clear(&app_for_main);
                debug!("Cleared current input field using delete shortcut follow-up");
            }
            Err(err) => error!("Failed to clear current input field: {}", err),
        })
    {
        error!("Failed to schedule current input clear: {:?}", err);
    }
}

#[cfg(target_os = "macos")]
fn delete_last_inserted_transcription(app: &AppHandle) -> bool {
    let Some(char_count) = take_valid_candidate(app) else {
        return false;
    };

    let app_for_main = app.clone();
    if let Err(err) =
        app.run_on_main_thread(move || match send_backspaces(&app_for_main, char_count) {
            Ok(()) => {
                crate::post_process_context::clear(&app_for_main);
                debug!(
                    "Deleted recent transcription insertion: {} chars",
                    char_count
                );
            }
            Err(err) => error!("Failed to delete recent transcription insertion: {}", err),
        })
    {
        error!(
            "Failed to schedule recent transcription deletion: {:?}",
            err
        );
        return false;
    }

    true
}

#[cfg(target_os = "macos")]
pub fn replace_recent_insertion(
    app: &AppHandle,
    previous_char_count: usize,
    replacement_text: String,
) -> Result<usize, String> {
    send_backspaces(app, previous_char_count)?;
    crate::utils::paste(replacement_text, app.clone())
}

#[cfg(not(target_os = "macos"))]
pub fn replace_recent_insertion(
    app: &AppHandle,
    _previous_char_count: usize,
    replacement_text: String,
) -> Result<usize, String> {
    crate::utils::paste(replacement_text, app.clone())
}

#[cfg(target_os = "macos")]
pub fn delete_recent_transcription(app: &AppHandle) {
    if take_valid_current_input_clear(app) {
        clear_current_input(app);
        return;
    }

    if delete_last_inserted_transcription(app) {
        arm_current_input_clear(app);
    } else {
        arm_current_input_clear(app);
    }
}

#[cfg(target_os = "macos")]
fn current_undo_hotkey(app: &AppHandle) -> Option<handy_keys::Hotkey> {
    let settings = crate::settings::get_settings(app);
    if !settings.recent_transcription_undo_enabled {
        clear_candidate(app);
        return None;
    }

    let binding = settings
        .bindings
        .get(DELETE_RECENT_TRANSCRIPTION_BINDING_ID)?;
    match binding.current_binding.parse::<handy_keys::Hotkey>() {
        Ok(hotkey) => Some(hotkey),
        Err(err) => {
            warn!(
                "Recent transcription undo shortcut '{}' is invalid: {}",
                binding.current_binding, err
            );
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn event_presses_hotkey(event: &handy_keys::KeyEvent, hotkey: handy_keys::Hotkey) -> bool {
    event.is_key_down && hotkey.modifiers.matches(event.modifiers) && hotkey.key == event.key
}

#[cfg(target_os = "macos")]
fn event_releases_hotkey(event: &handy_keys::KeyEvent, hotkey: handy_keys::Hotkey) -> bool {
    if event.is_key_down {
        return false;
    }

    match hotkey.key {
        Some(key) => event.key == Some(key),
        None => event
            .changed_modifier
            .map(|changed| hotkey.modifiers.contains(changed))
            .unwrap_or(false),
    }
}

#[cfg(target_os = "macos")]
pub fn start_shortcut_listener(app: &AppHandle) {
    let Some(state) = app.try_state::<RecentTranscriptionUndo>() else {
        warn!("RecentTranscriptionUndo state is not initialized");
        return;
    };

    if state.listener_started.swap(true, Ordering::SeqCst) {
        return;
    }

    let app_for_listener = app.clone();
    std::thread::spawn(move || {
        let listener = match handy_keys::KeyboardListener::new() {
            Ok(listener) => listener,
            Err(err) => {
                error!(
                    "Recent transcription undo shortcut listener failed to start: {}",
                    err
                );
                if let Some(state) = app_for_listener.try_state::<RecentTranscriptionUndo>() {
                    state.listener_started.store(false, Ordering::SeqCst);
                }
                return;
            }
        };

        let mut armed_hotkey: Option<handy_keys::Hotkey> = None;
        let mut used_with_other_input = false;

        loop {
            let event = match listener.recv_timeout(Duration::from_millis(50)) {
                Ok(event) => event,
                Err(handy_keys::Error::Timeout) => continue,
                Err(err) => {
                    error!(
                        "Recent transcription undo shortcut listener stopped: {}",
                        err
                    );
                    if let Some(state) = app_for_listener.try_state::<RecentTranscriptionUndo>() {
                        state.listener_started.store(false, Ordering::SeqCst);
                    }
                    break;
                }
            };

            if !has_pending_delete_action(&app_for_listener) {
                armed_hotkey = None;
                used_with_other_input = false;
                continue;
            }

            let Some(target_hotkey) = current_undo_hotkey(&app_for_listener) else {
                armed_hotkey = None;
                used_with_other_input = false;
                continue;
            };

            if let Some(armed) = armed_hotkey {
                if event_releases_hotkey(&event, armed) {
                    if !used_with_other_input {
                        delete_recent_transcription(&app_for_listener);
                    }
                    armed_hotkey = None;
                    used_with_other_input = false;
                } else if event.is_key_down && !event_presses_hotkey(&event, armed) {
                    used_with_other_input = true;
                }
            } else if event_presses_hotkey(&event, target_hotkey) {
                armed_hotkey = Some(target_hotkey);
                used_with_other_input = false;
            }
        }
    });
}

#[cfg(not(target_os = "macos"))]
pub fn start_shortcut_listener(_app: &AppHandle) {}
