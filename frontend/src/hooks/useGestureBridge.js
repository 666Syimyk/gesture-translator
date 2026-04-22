import { useEffect, useRef, useState } from "react";
import {
  buildGestureBridgeWsUrl,
  resetGestureBridge,
  startGestureBridge,
  stopGestureBridge,
} from "../api/gestureBridgeApi";
import { normalizeGesturePredictionMessage } from "../utils/gestureLabels";

const RECONNECT_DELAY_MS = 1000;
const PING_INTERVAL_MS = 3000;
const MAX_LOCAL_FRAME_BYTES = 900_000;
const TEXT_ENCODER =
  typeof TextEncoder !== "undefined" ? new TextEncoder() : null;

function buildPredictionSignature(payload) {
  if (!payload) {
    return "";
  }

  return [
    payload.label_id,
    payload.confidence,
    payload.stable ? 1 : 0,
    payload.debug?.tracking_ok ? 1 : 0,
    payload.debug?.hands_count ?? 0,
    payload.debug?.dominant_zone ?? "",
    payload.debug?.movement_type ?? "",
    payload.debug?.repeat_count ?? 0,
    payload.debug?.none_reason ?? "",
  ].join("|");
}

function buildStatusSignature(payload) {
  if (!payload) {
    return "";
  }

  return [
    payload.running ? 1 : 0,
    payload.ready ? 1 : 0,
    payload.bufferSize ?? payload.buffer_size ?? 0,
    (payload.trackingOk ?? payload.tracking_ok) ? 1 : 0,
    payload.frameInFlight ? 1 : 0,
    payload.dominant_zone ?? "",
    payload.movement_type ?? "",
    payload.repeat_count ?? 0,
    payload.lastError ?? "",
    payload.metrics?.framesDroppedBusy ?? 0,
    payload.metrics?.lastLatencyMs ?? 0,
  ].join("|");
}

function trimRecentTimestamps(items, now) {
  return [...items, now].filter((value) => now - value <= 1000);
}

export function useGestureBridge() {
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const pingTimerRef = useRef(null);
  const shouldStayConnectedRef = useRef(false);
  const workerStatusRef = useRef(null);
  const diagnosticsRef = useRef({
    framesSent: 0,
    sendFps: 0,
    wsMessages: 0,
    lastRoundtripMs: 0,
    avgRoundtripMs: 0,
    roundtripSamples: 0,
    lastPingLatencyMs: 0,
    avgPingLatencyMs: 0,
    pingSamples: 0,
    framesDroppedHidden: 0,
    framesDroppedSocket: 0,
    framesDroppedBusyLocal: 0,
    framesDroppedOversizeLocal: 0,
    framesDroppedByBackend: 0,
    lastFrameDropReason: "",
    lastPredictionAt: null,
    lastStateAt: null,
    lastPongAt: null,
  });
  const sentFrameTimestampsRef = useRef([]);
  const pendingFramesRef = useRef(new Map());
  const pendingPingsRef = useRef(new Map());
  const nextFrameIdRef = useRef(1);
  const nextPingIdRef = useRef(1);
  const lastPredictionSignatureRef = useRef("");
  const lastStatusSignatureRef = useRef("");

  const [connectionState, setConnectionState] = useState("disconnected");
  const [workerStatus, setWorkerStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [diagnostics, setDiagnostics] = useState(diagnosticsRef.current);
  const [error, setError] = useState("");

  function measurePayloadBytes(payload) {
    if (TEXT_ENCODER) {
      return TEXT_ENCODER.encode(String(payload || "")).length;
    }
    return String(payload || "").length;
  }

  function updateDiagnostics(patch) {
    const nextValue =
      typeof patch === "function" ? patch(diagnosticsRef.current) : { ...diagnosticsRef.current, ...patch };
    diagnosticsRef.current = nextValue;
    setDiagnostics(nextValue);
  }

  function clearLocalState() {
    lastPredictionSignatureRef.current = "";
    lastStatusSignatureRef.current = "";
    workerStatusRef.current = null;
    sentFrameTimestampsRef.current = [];
    pendingFramesRef.current.clear();
    pendingPingsRef.current.clear();
    setPrediction(null);
    setWorkerStatus(null);
    setError("");
    const nextDiagnostics = {
      framesSent: 0,
      sendFps: 0,
      wsMessages: 0,
      lastRoundtripMs: 0,
      avgRoundtripMs: 0,
      roundtripSamples: 0,
      lastPingLatencyMs: 0,
      avgPingLatencyMs: 0,
      pingSamples: 0,
      framesDroppedHidden: 0,
      framesDroppedSocket: 0,
      framesDroppedBusyLocal: 0,
      framesDroppedOversizeLocal: 0,
      framesDroppedByBackend: 0,
      lastFrameDropReason: "",
      lastPredictionAt: null,
      lastStateAt: null,
      lastPongAt: null,
    };
    diagnosticsRef.current = nextDiagnostics;
    setDiagnostics(nextDiagnostics);
  }

  function updateConnectionState(nextState) {
    setConnectionState(nextState);
  }

  function handleFrameAck(message) {
    const frameId = Number(message.client_frame_id ?? 0);
    if (!frameId || !pendingFramesRef.current.has(frameId)) {
      return;
    }

    const sentAt = pendingFramesRef.current.get(frameId);
    pendingFramesRef.current.delete(frameId);
    const roundtripMs = performance.now() - sentAt;
    updateDiagnostics((current) => {
      const roundtripSamples = current.roundtripSamples + 1;
      return {
        ...current,
        lastRoundtripMs: roundtripMs,
        avgRoundtripMs:
          current.avgRoundtripMs + (roundtripMs - current.avgRoundtripMs) / roundtripSamples,
        roundtripSamples,
        ...(message.type === "state" ? { lastStateAt: Date.now() } : {}),
        ...(message.type === "prediction" ? { lastPredictionAt: Date.now() } : {}),
      };
    });
  }

  function handlePong(message) {
    const pingId = Number(message.client_ping_id ?? 0);
    if (!pingId || !pendingPingsRef.current.has(pingId)) {
      updateDiagnostics((current) => ({
        ...current,
        lastPongAt: Date.now(),
      }));
      return;
    }

    const sentAt = pendingPingsRef.current.get(pingId);
    pendingPingsRef.current.delete(pingId);
    const pingLatencyMs = performance.now() - sentAt;
    updateDiagnostics((current) => {
      const pingSamples = current.pingSamples + 1;
      return {
        ...current,
        lastPingLatencyMs: pingLatencyMs,
        avgPingLatencyMs:
          current.avgPingLatencyMs + (pingLatencyMs - current.avgPingLatencyMs) / pingSamples,
        pingSamples,
        lastPongAt: Date.now(),
      };
    });
  }

  function startPingLoop() {
    if (pingTimerRef.current) {
      return;
    }

    pingTimerRef.current = window.setInterval(() => {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }

      const pingId = nextPingIdRef.current;
      nextPingIdRef.current += 1;
      pendingPingsRef.current.set(pingId, performance.now());
      socket.send(
        JSON.stringify({
          type: "ping",
          client_ping_id: pingId,
          client_sent_at_ms: Date.now(),
        }),
      );
    }, PING_INTERVAL_MS);
  }

  function stopPingLoop() {
    if (pingTimerRef.current) {
      window.clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }
  }

  function scheduleReconnect() {
    if (!shouldStayConnectedRef.current || reconnectTimerRef.current) {
      return;
    }

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connect();
    }, RECONNECT_DELAY_MS);
  }

  function connect() {
    if (
      socketRef.current &&
      (socketRef.current.readyState === WebSocket.OPEN ||
        socketRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return socketRef.current;
    }

    const socket = new WebSocket(buildGestureBridgeWsUrl());
    socketRef.current = socket;
    updateConnectionState("connecting");

    socket.onopen = () => {
      updateConnectionState("connected");
      setError("");
      startPingLoop();
      console.info("[gesture-bridge-ui] websocket connected");
    };

    socket.onmessage = (event) => {
      updateDiagnostics((current) => ({
        ...current,
        wsMessages: current.wsMessages + 1,
      }));

      try {
        const message = normalizeGesturePredictionMessage(JSON.parse(event.data));

        if (message.type === "error") {
          setError(message.message || "Gesture bridge error");
          return;
        }

        if (message.type === "frame_drop") {
          const frameId = Number(message.client_frame_id ?? 0);
          if (frameId && pendingFramesRef.current.has(frameId)) {
            pendingFramesRef.current.delete(frameId);
          }
          updateDiagnostics((current) => ({
            ...current,
            framesDroppedByBackend: current.framesDroppedByBackend + 1,
            lastFrameDropReason: message.reason || "unknown",
          }));
          return;
        }

        if (message.type === "pong") {
          handlePong(message);
          return;
        }

        if (message.type === "prediction") {
          handleFrameAck(message);
          const signature = buildPredictionSignature(message);
          if (signature !== lastPredictionSignatureRef.current) {
            lastPredictionSignatureRef.current = signature;
            setPrediction(message);
          }
          return;
        }

        if (message.type === "ready") {
          setError("");
          const nextStatus = {
            ...(workerStatusRef.current || {}),
            running: true,
            ready: true,
            modelMode: message.model_mode,
            sequenceLength: message.sequence_length,
            minSequenceLength: message.min_sequence_length,
          };
          const signature = buildStatusSignature(nextStatus);
          if (signature !== lastStatusSignatureRef.current) {
            lastStatusSignatureRef.current = signature;
            workerStatusRef.current = nextStatus;
            setWorkerStatus(nextStatus);
          }
          return;
        }

        if (message.type === "status" || message.type === "state") {
          if (message.type === "state") {
            handleFrameAck(message);
          }

          const nextStatus = {
            ...(workerStatusRef.current || {}),
            ...message,
          };
          if (
            (nextStatus.ready || nextStatus.running) &&
            !nextStatus.lastError
          ) {
            setError("");
          }
          const signature = buildStatusSignature(nextStatus);
          if (signature !== lastStatusSignatureRef.current) {
            lastStatusSignatureRef.current = signature;
            workerStatusRef.current = nextStatus;
            setWorkerStatus(nextStatus);
          }
        }
      } catch {
        setError("Не удалось прочитать ответ gesture bridge");
      }
    };

    socket.onerror = () => {
      setError("Ошибка соединения с gesture bridge");
    };

    socket.onclose = () => {
      socketRef.current = null;
      stopPingLoop();
      updateConnectionState("disconnected");
      console.info("[gesture-bridge-ui] websocket disconnected");
      if (shouldStayConnectedRef.current) {
        scheduleReconnect();
      }
    };

    return socket;
  }

  async function start() {
    shouldStayConnectedRef.current = true;
    connect();
    const status = await startGestureBridge();
    const signature = buildStatusSignature(status);
    lastStatusSignatureRef.current = signature;
    workerStatusRef.current = status;
    setWorkerStatus(status);
    return status;
  }

  async function stop() {
    shouldStayConnectedRef.current = false;

    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    stopPingLoop();

    try {
      await stopGestureBridge();
    } finally {
      socketRef.current?.close();
      socketRef.current = null;
      updateConnectionState("disconnected");
      clearLocalState();
    }
  }

  async function reset() {
    const status = await resetGestureBridge();
    lastPredictionSignatureRef.current = "";
    setPrediction(null);
    setError("");
    pendingFramesRef.current.clear();
    const signature = buildStatusSignature(status);
    lastStatusSignatureRef.current = signature;
    workerStatusRef.current = status;
    setWorkerStatus(status);
    return status;
  }

  function sendFrame({ ts, imageBase64 }) {
    if (document.hidden) {
      updateDiagnostics((current) => ({
        ...current,
        framesDroppedHidden: current.framesDroppedHidden + 1,
      }));
      return false;
    }

    if (measurePayloadBytes(imageBase64) > MAX_LOCAL_FRAME_BYTES) {
      updateDiagnostics((current) => ({
        ...current,
        framesDroppedOversizeLocal: current.framesDroppedOversizeLocal + 1,
        lastFrameDropReason: "oversize_local",
      }));
      return false;
    }

    if (pendingFramesRef.current.size > 0) {
      updateDiagnostics((current) => ({
        ...current,
        framesDroppedBusyLocal: current.framesDroppedBusyLocal + 1,
        lastFrameDropReason: "pending_ack",
      }));
      return false;
    }

    if (workerStatusRef.current?.frameInFlight) {
      updateDiagnostics((current) => ({
        ...current,
        framesDroppedBusyLocal: current.framesDroppedBusyLocal + 1,
        lastFrameDropReason: "busy_local",
      }));
      return false;
    }

    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      updateDiagnostics((current) => ({
        ...current,
        framesDroppedSocket: current.framesDroppedSocket + 1,
        lastFrameDropReason: "socket_unavailable",
      }));
      return false;
    }

    const clientFrameId = nextFrameIdRef.current;
    nextFrameIdRef.current += 1;
    pendingFramesRef.current.set(clientFrameId, performance.now());

    if (pendingFramesRef.current.size > 2) {
      const oldestFrameId = pendingFramesRef.current.keys().next().value;
      pendingFramesRef.current.delete(oldestFrameId);
    }

    const now = performance.now();
    sentFrameTimestampsRef.current = trimRecentTimestamps(sentFrameTimestampsRef.current, now);

    updateDiagnostics((current) => ({
      ...current,
      framesSent: current.framesSent + 1,
      sendFps: sentFrameTimestampsRef.current.length,
    }));

    socket.send(
      JSON.stringify({
        type: "frame",
        ts,
        image_b64: imageBase64,
        client_frame_id: clientFrameId,
        client_sent_at_ms: Date.now(),
      }),
    );
    return true;
  }

  useEffect(() => {
    return () => {
      shouldStayConnectedRef.current = false;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      stopPingLoop();
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  return {
    connectionState,
    connected: connectionState === "connected",
    workerReady: Boolean(workerStatus?.ready),
    status: workerStatus,
    prediction,
    diagnostics,
    error,
    start,
    stop,
    reset,
    sendFrame,
    clearLocalState,
  };
}
