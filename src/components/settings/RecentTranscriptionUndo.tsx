import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../ui/Slider";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface RecentTranscriptionUndoProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const RecentTranscriptionUndo: React.FC<RecentTranscriptionUndoProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("recent_transcription_undo_enabled") ?? true;
    const undoWindowMs = getSetting("recent_transcription_undo_window_ms") ?? 5000;

    return (
      <>
        <ToggleSwitch
          checked={enabled}
          onChange={(enabled) =>
            updateSetting("recent_transcription_undo_enabled", enabled)
          }
          isUpdating={isUpdating("recent_transcription_undo_enabled")}
          label={t("settings.advanced.recentTranscriptionUndo.label")}
          description={t(
            "settings.advanced.recentTranscriptionUndo.description",
          )}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />
        {enabled && (
          <Slider
            value={undoWindowMs}
            onChange={(value) =>
              updateSetting("recent_transcription_undo_window_ms", value)
            }
            min={500}
            max={60000}
            step={500}
            disabled={isUpdating("recent_transcription_undo_window_ms")}
            label={`${t("settings.advanced.recentTranscriptionUndo.label")} 시간`}
            description="직전 전사를 삭제할 수 있는 시간입니다. 두 번 눌러 전체 삭제하는 동작은 이 시간 제한을 받지 않습니다."
            descriptionMode={descriptionMode}
            grouped={grouped}
            formatValue={(value) => `${(value / 1000).toFixed(1)}s`}
          />
        )}
      </>
    );
  });
