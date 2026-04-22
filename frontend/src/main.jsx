import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

function detectLikelyPhoneViewport() {
  if (typeof window === "undefined") {
    return false;
  }

  const narrowViewport = window.matchMedia("(max-width: 900px)").matches;
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const phoneUserAgent =
    /Android|iPhone|iPad|iPod|Mobile|Windows Phone/i.test(
      String(window.navigator?.userAgent ?? ""),
    );

  return narrowViewport && (coarsePointer || phoneUserAgent);
}

const searchParams = new URLSearchParams(window.location.search)
const normalizedPath = window.location.pathname.replace(/\/+$/, "").toLowerCase()
const uiParam = String(searchParams.get("ui") ?? "").toLowerCase()
const uiMode =
  uiParam === "admin" || normalizedPath.endsWith("/admin")
    ? "admin"
    : uiParam === "user" || normalizedPath.endsWith("/user") || detectLikelyPhoneViewport()
      ? "user"
      : "admin"

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App uiMode={uiMode} />
  </StrictMode>,
)

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
