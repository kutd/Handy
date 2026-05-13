import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  MicrophoneIcon,
  TranscriptionIcon,
  CancelIcon,
} from "../components/icons";
import "./RecordingOverlay.css";
import { commands } from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import type { InterimTranscriptionEvent } from "@/lib/types/events";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState = "recording" | "transcribing" | "processing";

type InterimSegment = {
  text: string;
  sampleStart: number;
  sampleEnd: number;
};

const MAX_INTERIM_DISPLAY_CHARS = 900;
const MAX_INTERIM_SEGMENTS = 80;

const normalizePreviewText = (text: string) => text.replace(/\s+/g, " ").trim();

const normalizeForOverlap = (text: string) =>
  text.toLocaleLowerCase().replace(/[\s"'“”‘’.,!?;:()[\]{}<>，。！？、]/g, "");

const clampInterimDisplay = (text: string) => {
  if (text.length <= MAX_INTERIM_DISPLAY_CHARS) {
    return text;
  }

  const clipped = text.slice(text.length - MAX_INTERIM_DISPLAY_CHARS);
  const firstSpace = clipped.search(/\s/);
  return firstSpace > 0 ? clipped.slice(firstSpace + 1) : clipped;
};

const joinInterimText = (segments: InterimSegment[]) =>
  clampInterimDisplay(
    normalizePreviewText(segments.map((segment) => segment.text).join(" ")),
  );

const hasSampleRange = (
  event: InterimTranscriptionEvent,
): event is InterimTranscriptionEvent & {
  sample_start: number;
  sample_end: number;
} =>
  Number.isFinite(event.sample_start) &&
  Number.isFinite(event.sample_end) &&
  Number(event.sample_end) > Number(event.sample_start);

const trimSegmentBefore = (
  segment: InterimSegment,
  sampleStart: number,
): InterimSegment | null => {
  if (segment.sampleEnd <= sampleStart) {
    return segment;
  }
  if (
    segment.sampleStart >= sampleStart ||
    segment.sampleEnd <= segment.sampleStart
  ) {
    return null;
  }

  const ratio =
    (sampleStart - segment.sampleStart) /
    (segment.sampleEnd - segment.sampleStart);
  const estimatedChars = Math.floor(segment.text.length * ratio);
  const prefix = segment.text
    .slice(0, estimatedChars)
    .replace(/\s+\S*$/, "")
    .trim();

  return prefix
    ? {
        text: prefix,
        sampleStart: segment.sampleStart,
        sampleEnd: sampleStart,
      }
    : null;
};

const mergeSegmentByRange = (
  segments: InterimSegment[],
  incoming: InterimSegment,
  replaceExisting: boolean,
) => {
  if (replaceExisting) {
    const next = segments
      .flatMap((segment) => {
        if (segment.sampleEnd <= incoming.sampleStart) {
          return [segment];
        }
        if (segment.sampleStart >= incoming.sampleEnd) {
          return [segment];
        }

        const prefix = trimSegmentBefore(segment, incoming.sampleStart);
        return prefix ? [prefix] : [];
      })
      .concat(incoming)
      .sort(
        (a, b) => a.sampleStart - b.sampleStart || a.sampleEnd - b.sampleEnd,
      );

    return next.slice(-MAX_INTERIM_SEGMENTS);
  }

  const next = segments
    .filter(
      (segment) =>
        segment.sampleEnd <= incoming.sampleStart ||
        segment.sampleStart >= incoming.sampleEnd,
    )
    .concat(incoming)
    .sort((a, b) => a.sampleStart - b.sampleStart || a.sampleEnd - b.sampleEnd);

  return next.slice(-MAX_INTERIM_SEGMENTS);
};

const mergeTextByOverlap = (previousText: string, replacementText: string) => {
  const previous = normalizePreviewText(previousText);
  const replacement = normalizePreviewText(replacementText);

  if (!previous || previous === replacement || previous.endsWith(replacement)) {
    return previous || replacement;
  }
  if (replacement.startsWith(previous)) {
    return replacement;
  }

  const maxOverlap = Math.min(120, previous.length, replacement.length);
  for (let length = maxOverlap; length >= 3; length -= 1) {
    const previousTail = previous.slice(previous.length - length);
    const replacementHead = replacement.slice(0, length);
    if (
      normalizeForOverlap(previousTail) &&
      normalizeForOverlap(previousTail) === normalizeForOverlap(replacementHead)
    ) {
      return normalizePreviewText(
        `${previous.slice(0, previous.length - length)} ${replacement}`,
      );
    }
  }

  return replacement;
};

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  const [interimText, setInterimText] = useState("");
  const [levels, setLevels] = useState<number[]>(Array(16).fill(0));
  const smoothedLevelsRef = useRef<number[]>(Array(16).fill(0));
  const interimSegmentsRef = useRef<InterimSegment[]>([]);
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const setupEventListeners = async () => {
      // Listen for show-overlay event from Rust
      const unlistenShow = await listen("show-overlay", async (event) => {
        // Sync language from settings each time overlay is shown
        await syncLanguageFromSettings();
        const overlayState = event.payload as OverlayState;
        setState(overlayState);
        if (overlayState === "recording") {
          interimSegmentsRef.current = [];
          setInterimText("");
        }
        setIsVisible(true);
      });

      // Listen for hide-overlay event from Rust
      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
        interimSegmentsRef.current = [];
        setInterimText("");
      });

      // Listen for mic-level updates
      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const newLevels = event.payload as number[];

        // Apply smoothing to reduce jitter
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = newLevels[i] || 0;
          return prev * 0.7 + target * 0.3; // Smooth transition
        });

        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed.slice(0, 9));
      });

      const unlistenInterim = await listen<InterimTranscriptionEvent>(
        "interim-transcription",
        (event) => {
          const text = normalizePreviewText(event.payload.text);
          if (!text) {
            return;
          }

          if (hasSampleRange(event.payload)) {
            interimSegmentsRef.current = mergeSegmentByRange(
              interimSegmentsRef.current,
              {
                text,
                sampleStart: event.payload.sample_start,
                sampleEnd: event.payload.sample_end,
              },
              event.payload.replace_existing,
            );
            setInterimText(joinInterimText(interimSegmentsRef.current));
            return;
          }

          setInterimText((previousText) =>
            clampInterimDisplay(
              event.payload.replace_existing
                ? mergeTextByOverlap(previousText, text)
                : normalizePreviewText(
                    previousText ? `${previousText} ${text}` : text,
                  ),
            ),
          );
        },
      );

      // Cleanup function
      return () => {
        unlistenShow();
        unlistenHide();
        unlistenLevel();
        unlistenInterim();
      };
    };

    setupEventListeners();
  }, []);

  const getIcon = () => {
    if (state === "recording") {
      return <MicrophoneIcon />;
    } else {
      return <TranscriptionIcon />;
    }
  };

  return (
    <div
      dir={direction}
      className={`recording-overlay ${isVisible ? "fade-in" : ""} ${
        interimText ? "has-interim" : ""
      }`}
    >
      <div className="overlay-left">{getIcon()}</div>

      <div className="overlay-middle">
        {state === "recording" && (
          <div className="recording-content">
            <div className="bars-container">
              {levels.map((v, i) => (
                <div
                  key={i}
                  className="bar"
                  style={{
                    height: `${Math.min(20, 4 + Math.pow(v, 0.7) * 16)}px`, // Cap at 20px max height
                    transition: "height 60ms ease-out, opacity 120ms ease-out",
                    opacity: Math.max(0.2, v * 1.7), // Minimum opacity for visibility
                  }}
                />
              ))}
            </div>
            {interimText && <div className="interim-text">{interimText}</div>}
          </div>
        )}
        {state === "transcribing" && (
          <div className="transcribing-text">{t("overlay.transcribing")}</div>
        )}
        {state === "processing" && (
          <div className="transcribing-text">{t("overlay.processing")}</div>
        )}
      </div>

      <div className="overlay-right">
        {state === "recording" && (
          <div
            className="cancel-button"
            onClick={() => {
              commands.cancelOperation();
            }}
          >
            <CancelIcon />
          </div>
        )}
      </div>
    </div>
  );
};

export default RecordingOverlay;
