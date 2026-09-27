import { useEffect, useReducer } from "react";

import {
  clearStoredState,
  conditionFromLocation,
  createInitialState,
  gameReducer,
  loadStoredState,
  participantCodeFromLocation,
  saveState,
} from "./gameState";
import type { GameState } from "./types";

function initialize(): GameState {
  const participantCode = participantCodeFromLocation(window.location.search);
  const stored = loadStoredState();
  if (stored && stored.participantCode === participantCode) return stored;
  if (stored) clearStoredState();
  return createInitialState(
    conditionFromLocation(window.location.search),
    participantCode,
  );
}

export function useSession() {
  const [state, dispatch] = useReducer(gameReducer, undefined, initialize);

  useEffect(() => {
    saveState(state);
  }, [state]);

  function reset(): void {
    clearStoredState();
    window.location.reload();
  }

  function exportSession(): void {
    const exportState = state.generatedPattern
      ? {
          ...state,
          generatedPattern: {
            ...state.generatedPattern,
            imageDataUrl: "[omitted-from-research-log]",
          },
        }
      : state;
    const blob = new Blob([JSON.stringify(exportState, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `lanternquest-${state.sessionId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return { state, dispatch, reset, exportSession };
}
