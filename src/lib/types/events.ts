export interface ModelStateEvent {
  event_type: string;
  model_id?: string;
  model_name?: string;
  error?: string;
}

export interface RecordingErrorEvent {
  error_type: string;
  detail?: string;
}

export interface InterimTranscriptionEvent {
  text: string;
  sample_count: number;
  sample_start?: number;
  sample_end?: number;
  replace_existing: boolean;
}
