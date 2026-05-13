import React from "react";
import { useTranslation } from "react-i18next";
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

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(enabled) =>
          updateSetting("recent_transcription_undo_enabled", enabled)
        }
        isUpdating={isUpdating("recent_transcription_undo_enabled")}
        label={t("settings.advanced.recentTranscriptionUndo.label")}
        description={t("settings.advanced.recentTranscriptionUndo.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  });
