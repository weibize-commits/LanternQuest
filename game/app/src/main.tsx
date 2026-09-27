import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import VRFigureCapture from "./VRFigureCapture";
import "./styles.css";

const RootComponent = new URLSearchParams(globalThis.location.search).get("vrFigure") === "1"
  ? VRFigureCapture
  : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootComponent />
  </StrictMode>,
);
