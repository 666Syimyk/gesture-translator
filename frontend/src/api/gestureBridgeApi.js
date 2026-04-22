import { buildApiUrl, request } from "./apiClient";

export function buildGestureBridgeWsUrl() {
  const wsPath = buildApiUrl("/api/gestures/ws");

  if (/^https?:\/\//i.test(wsPath)) {
    return wsPath.replace(/^http/i, "ws");
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${wsPath}`;
}

export function fetchGestureBridgeStatus() {
  return request("/api/gestures/status");
}

export function startGestureBridge() {
  return request("/api/gestures/start", {
    method: "POST",
  });
}

export function stopGestureBridge() {
  return request("/api/gestures/stop", {
    method: "POST",
  });
}

export function resetGestureBridge() {
  return request("/api/gestures/reset", {
    method: "POST",
  });
}
