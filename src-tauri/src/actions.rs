#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";
const POST_PROCESS_OPERATION_FIELD: &str = "operation";
const POST_PROCESS_OPERATION_INSERT: &str = "insert";
const POST_PROCESS_OPERATION_REPLACE_PREVIOUS: &str = "replace_previous";
const POST_PROCESS_CONTEXT_WINDOW: Duration = Duration::from_secs(30);

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

fn recent_context_instruction(settings: &AppSettings) -> String {
    let prompt = settings.post_process_context_prompt.trim();
    if prompt.is_empty() {
        crate::settings::default_post_process_context_prompt()
    } else {
        prompt.to_string()
    }
}

#[derive(Clone, Debug)]
struct PostProcessResult {
    text: String,
    replace_previous: bool,
}

impl PostProcessResult {
    fn insert(text: String) -> Self {
        Self {
            text,
            replace_previous: false,
        }
    }

    fn with_operation(text: String, operation: Option<&str>) -> Self {
        Self {
            text,
            replace_previous: matches!(
                operation,
                Some(value) if value == POST_PROCESS_OPERATION_REPLACE_PREVIOUS
            ),
        }
    }
}

fn previous_inserted_text(context: &crate::post_process_context::RecentPostProcessContext) -> &str {
    context
        .post_processed_text
        .as_deref()
        .unwrap_or(&context.final_text)
}

fn format_recent_context(
    context: &crate::post_process_context::RecentPostProcessContext,
) -> String {
    format!(
        "P_raw:\n{}\n\nP:\n{}",
        context.raw_transcription,
        previous_inserted_text(context)
    )
}

fn build_contextual_user_content(
    transcription: &str,
    context: Option<&crate::post_process_context::RecentPostProcessContext>,
) -> String {
    match context {
        Some(context) => format!(
            "{}\n\nC:\n{}",
            format_recent_context(context),
            transcription
        ),
        None => transcription.to_string(),
    }
}

fn build_contextual_system_prompt(prompt: &str, context_prompt: &str) -> String {
    let cleanup_prompt = build_system_prompt(prompt);
    if cleanup_prompt.is_empty() {
        return context_prompt.to_string();
    }

    format!(
        "{context_prompt}\n\nCleanup rules for the `{TRANSCRIPTION_FIELD}` value only:\n{cleanup_prompt}\n\nIf the cleanup rules ask for plain text, treat that as applying inside the JSON `{TRANSCRIPTION_FIELD}` value. Always return the required JSON object."
    )
}

fn build_legacy_post_process_prompt(
    prompt: &str,
    transcription: &str,
    context: Option<&crate::post_process_context::RecentPostProcessContext>,
    context_prompt: &str,
) -> String {
    match context {
        Some(context) => format!(
            "{}\n\n{}\n\nC:\n{}",
            context_prompt,
            format_recent_context(context),
            transcription
        ),
        None => prompt.replace("${output}", transcription),
    }
}

fn build_legacy_context_decision_prompt(
    prompt_template: &str,
    transcription: &str,
    context: &crate::post_process_context::RecentPostProcessContext,
    context_prompt: &str,
) -> String {
    let cleanup_prompt = build_system_prompt(prompt_template);
    let cleanup_block = if cleanup_prompt.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nCleanup rules for the `{TRANSCRIPTION_FIELD}` value only:\n{cleanup_prompt}\n\nIf the cleanup rules ask for plain text, treat that as applying inside the JSON `{TRANSCRIPTION_FIELD}` value. Always return the required JSON object."
        )
    };

    format!(
        "{}{}\n\n{}\n\nC:\n{}",
        context_prompt,
        cleanup_block,
        format_recent_context(context),
        transcription
    )
}

fn json_object_candidate(content: &str) -> &str {
    let trimmed = content.trim();
    match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if start <= end => &trimmed[start..=end],
        _ => trimmed,
    }
}

fn parse_post_process_json_result(content: &str) -> Option<PostProcessResult> {
    let json = serde_json::from_str::<serde_json::Value>(json_object_candidate(content)).ok()?;
    let transcription = json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())?;
    let operation = json
        .get(POST_PROCESS_OPERATION_FIELD)
        .and_then(|value| value.as_str());

    Some(PostProcessResult::with_operation(
        strip_invisible_chars(transcription),
        operation,
    ))
}

async fn post_process_transcription(
    settings: &AppSettings,
    transcription: &str,
    recent_context: Option<&crate::post_process_context::RecentPostProcessContext>,
) -> Option<PostProcessResult> {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let context_prompt = recent_context_instruction(settings);

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = if recent_context.is_some() {
            build_contextual_system_prompt(&prompt, &context_prompt)
        } else {
            build_system_prompt(&prompt)
        };
        let user_content = build_contextual_user_content(transcription, recent_context);

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                let (apple_system_prompt, apple_user_content) = if let Some(context) =
                    recent_context
                {
                    (
                            "Follow the user request exactly. When asked to return JSON, return only valid JSON with no markdown or explanation.".to_string(),
                            build_legacy_context_decision_prompt(
                                &prompt,
                                transcription,
                                context,
                                &context_prompt,
                            ),
                        )
                } else {
                    (system_prompt.clone(), user_content.clone())
                };
                return match apple_intelligence::process_text_with_system_prompt(
                    &apple_system_prompt,
                    &apple_user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            if recent_context.is_some() {
                                parse_post_process_json_result(&result)
                                    .or_else(|| Some(PostProcessResult::insert(result)))
                            } else {
                                Some(PostProcessResult::insert(result))
                            }
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "For ordinary dictation, the cleaned new text to insert. For an edit request about previous inserted text, the complete replacement text for that previous insertion."
                },
                (POST_PROCESS_OPERATION_FIELD): {
                    "type": "string",
                    "enum": [
                        POST_PROCESS_OPERATION_INSERT,
                        POST_PROCESS_OPERATION_REPLACE_PREVIOUS
                    ],
                    "description": "Default to insert. Use replace_previous only when the current transcription clearly asks to edit, fix, replace, delete, or correct the recent previous text that Handy inserted into the active input field."
                }
            },
            "required": [TRANSCRIPTION_FIELD, POST_PROCESS_OPERATION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            reasoning_effort.clone(),
            reasoning.clone(),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                if let Some(result) = parse_post_process_json_result(&content) {
                    debug!(
                        "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                        provider.id,
                        result.text.len()
                    );
                    return Some(result);
                }

                error!("Structured output response missing expected JSON fields");
                return Some(PostProcessResult::insert(strip_invisible_chars(&content)));
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let legacy_expects_json = recent_context.is_some();
    let processed_prompt = if let Some(context) = recent_context {
        build_legacy_context_decision_prompt(&prompt, transcription, context, &context_prompt)
    } else {
        build_legacy_post_process_prompt(&prompt, transcription, recent_context, &context_prompt)
    };
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        reasoning_effort,
        reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            if legacy_expects_json {
                parse_post_process_json_result(&content)
                    .or_else(|| Some(PostProcessResult::insert(content)))
            } else {
                Some(PostProcessResult::insert(content))
            }
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    pub replace_previous_char_count: Option<usize>,
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    process_transcription_output_with_context(app, transcription, post_process, true).await
}

pub(crate) async fn process_transcription_output_with_context(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    use_recent_context: bool,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;
    let mut replace_previous_char_count: Option<usize> = None;

    if let Some(converted_text) = maybe_convert_chinese_variant(&settings, transcription).await {
        final_text = converted_text;
    }

    if post_process {
        let recent_context = if use_recent_context {
            crate::post_process_context::recent(app, POST_PROCESS_CONTEXT_WINDOW)
        } else {
            None
        };

        if let Some(processed) =
            post_process_transcription(&settings, &final_text, recent_context.as_ref()).await
        {
            let should_replace_previous = recent_context.as_ref().is_some_and(|context| {
                context.inserted_char_count > 0 && processed.replace_previous
            });

            if should_replace_previous {
                replace_previous_char_count = recent_context
                    .as_ref()
                    .map(|context| context.inserted_char_count);
            }

            post_processed_text = Some(processed.text.clone());
            final_text = processed.text;

            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
        replace_previous_char_count,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            crate::interim_transcription::start_session(app);
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());
        let mut final_transcription_intent = Some(tm.reserve_final_transcription());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);
        crate::interim_transcription::finish_session(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    drop(final_transcription_intent.take());
                    debug!("Recording produced no audio samples; skipping persistence");
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe_with_intent(
                        samples,
                        final_transcription_intent
                            .take()
                            .unwrap_or_else(|| tm.reserve_final_transcription()),
                    );

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            let processed =
                                process_transcription_output(&ah, &transcription, post_process)
                                    .await;
                            let raw_transcription_for_context = transcription.clone();

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription.clone(),
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let post_processed_for_context =
                                    processed.post_processed_text.clone();
                                let final_text = processed.final_text;
                                let final_text_for_context = final_text.clone();
                                let replace_previous_char_count =
                                    processed.replace_previous_char_count;
                                ah.run_on_main_thread(move || {
                                    let paste_result = if let Some(previous_char_count) =
                                        replace_previous_char_count
                                    {
                                        crate::recent_transcription_undo::replace_recent_insertion(
                                            &ah_clone,
                                            previous_char_count,
                                            final_text,
                                        )
                                    } else {
                                        utils::paste(final_text, ah_clone.clone())
                                    };

                                    match paste_result {
                                        Ok(inserted_char_count) => {
                                            debug!(
                                                "Text pasted successfully in {:?}",
                                                paste_time.elapsed()
                                            );
                                            crate::post_process_context::record(
                                                &ah_clone,
                                                raw_transcription_for_context,
                                                final_text_for_context,
                                                post_processed_for_context,
                                                inserted_char_count,
                                            );
                                        }
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            debug!("Global Shortcut Transcription error: {}", err);
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                drop(final_transcription_intent.take());
                debug!("No samples retrieved from recording stop");
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_replace_previous_operation_from_json() {
        let result = parse_post_process_json_result(
            r#"{"operation":"replace_previous","transcription":"이전 입력을 고친 문장입니다."}"#,
        )
        .expect("JSON result should parse");

        assert!(result.replace_previous);
        assert_eq!(result.text, "이전 입력을 고친 문장입니다.");
    }

    #[test]
    fn parses_json_even_when_wrapped_in_markdown() {
        let result = parse_post_process_json_result(
            "```json\n{\"operation\":\"insert\",\"transcription\":\"새 문장입니다.\"}\n```",
        )
        .expect("wrapped JSON result should parse");

        assert!(!result.replace_previous);
        assert_eq!(result.text, "새 문장입니다.");
    }

    #[test]
    fn contextual_system_prompt_preserves_selected_cleanup_prompt() {
        let prompt = "Clean this transcript:\n- Add punctuation.\n\nTranscript:\n${output}";
        let context_prompt = "Return JSON with operation and transcription. Default to insert.";

        let result = build_contextual_system_prompt(prompt, context_prompt);

        assert!(result.contains(context_prompt));
        assert!(result.contains("Clean this transcript"));
        assert!(result.contains("Add punctuation"));
        assert!(result.contains("Always return the required JSON object"));
    }
}
