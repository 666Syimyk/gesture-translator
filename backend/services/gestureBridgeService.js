import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import readline from "node:readline";

const DEFAULT_MIN_FRAME_INTERVAL_MS = 95;
const DEFAULT_MAX_FRAME_PAYLOAD_BYTES = 700_000;
const HEARTBEAT_INTERVAL_MS = 5000;
const HEARTBEAT_TIMEOUT_MS = 12000;
const GESTURE_LABEL_RU_BY_ID = Object.freeze({
  none: "none",
  privet: "\u041f\u0440\u0438\u0432\u0435\u0442",
  poka: "\u041f\u043e\u043a\u0430",
  ya: "\u042f",
  ty: "\u0422\u044b",
  muzhchina: "\u041c\u0443\u0436\u0447\u0438\u043d\u0430",
  zhenshchina: "\u0416\u0435\u043d\u0449\u0438\u043d\u0430",
  bolshoy: "\u0411\u043e\u043b\u044c\u0448\u043e\u0439",
  malenkiy: "\u041c\u0430\u043b\u0435\u043d\u044c\u043a\u0438\u0439",
  krasivyy: "\u041a\u0440\u0430\u0441\u0438\u0432\u044b\u0439",
  spasibo: "\u0421\u043f\u0430\u0441\u0438\u0431\u043e",
  est: "\u0415\u0441\u0442\u044c",
});
const GESTURE_LABEL_ID_BY_RU = Object.freeze(
  Object.fromEntries(Object.entries(GESTURE_LABEL_RU_BY_ID).map(([labelId, labelRu]) => [labelRu, labelId])),
);
const GESTURE_LABEL_ID_BY_RU_LOWER = Object.freeze(
  Object.fromEntries(
    Object.entries(GESTURE_LABEL_ID_BY_RU).map(([labelRu, labelId]) => [labelRu.toLowerCase(), labelId]),
  ),
);
const GESTURE_LABEL_ALIASES = Object.freeze({
  krasiviy: "krasivyy",
});

function resolveGestureLabelId(value) {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "none";
  }
  const lowerNormalized = normalized.toLowerCase();
  if (normalized in GESTURE_LABEL_RU_BY_ID) {
    return normalized;
  }
  if (lowerNormalized in GESTURE_LABEL_RU_BY_ID) {
    return lowerNormalized;
  }
  return (
    GESTURE_LABEL_ALIASES[lowerNormalized] ??
    GESTURE_LABEL_ID_BY_RU[normalized] ??
    GESTURE_LABEL_ID_BY_RU_LOWER[lowerNormalized] ??
    "none"
  );
}

function resolveGestureLabelRu(labelId, fallback = "") {
  const normalized = resolveGestureLabelId(labelId);
  if (normalized && normalized in GESTURE_LABEL_RU_BY_ID) {
    return GESTURE_LABEL_RU_BY_ID[normalized];
  }
  const rawFallback = String(fallback ?? "").trim();
  return rawFallback || "none";
}

function normalizeGestureRankedItem(item) {
  if (!item || typeof item !== "object") {
    return item;
  }

  const labelId = resolveGestureLabelId(item.label_id ?? item.labelId ?? item.label_ru ?? item.labelRu);
  const labelRu = resolveGestureLabelRu(labelId, item.label_ru ?? item.labelRu);
  return {
    ...item,
    label_id: labelId,
    label_ru: labelRu,
  };
}

function normalizeGestureBridgeMessage(message) {
  if (!message || typeof message !== "object" || message.type !== "prediction") {
    return message;
  }

  const labelId = resolveGestureLabelId(message.label_id ?? message.labelId ?? message.label_ru ?? message.labelRu);
  const topLabelId = resolveGestureLabelId(
    message.top_label_id ?? message.topLabelId ?? message.top_label_ru ?? message.topLabelRu,
  );
  const nextDebug =
    message.debug && typeof message.debug === "object"
      ? {
          ...message.debug,
          rule_score_summary:
            message.debug.rule_score_summary && typeof message.debug.rule_score_summary === "object"
              ? {
                  ...message.debug.rule_score_summary,
                  spatial_top: Array.isArray(message.debug.rule_score_summary.spatial_top)
                    ? message.debug.rule_score_summary.spatial_top.map(normalizeGestureRankedItem)
                    : message.debug.rule_score_summary.spatial_top,
                  temporal_top: Array.isArray(message.debug.rule_score_summary.temporal_top)
                    ? message.debug.rule_score_summary.temporal_top.map(normalizeGestureRankedItem)
                    : message.debug.rule_score_summary.temporal_top,
                }
              : message.debug.rule_score_summary,
        }
      : message.debug;

  return {
    ...message,
    label_id: labelId,
    label_ru: resolveGestureLabelRu(labelId, message.label_ru ?? message.labelRu),
    top_label_id: topLabelId,
    top_label_ru: resolveGestureLabelRu(topLabelId, message.top_label_ru ?? message.topLabelRu),
    top_final: Array.isArray(message.top_final) ? message.top_final.map(normalizeGestureRankedItem) : message.top_final,
    top_rules: Array.isArray(message.top_rules) ? message.top_rules.map(normalizeGestureRankedItem) : message.top_rules,
    top_model: Array.isArray(message.top_model) ? message.top_model.map(normalizeGestureRankedItem) : message.top_model,
    debug: nextDebug,
  };
}

export class GestureBridgeService extends EventEmitter {
  constructor({
    pythonBin,
    projectRoot,
    workerModule = "gesture_translator.server",
    minFrameIntervalMs = DEFAULT_MIN_FRAME_INTERVAL_MS,
    maxFramePayloadBytes = DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
  }) {
    super();
    this.pythonBin = pythonBin;
    this.projectRoot = projectRoot;
    this.workerModule = workerModule;
    this.minFrameIntervalMs = Math.max(20, Number(minFrameIntervalMs) || DEFAULT_MIN_FRAME_INTERVAL_MS);
    this.maxFramePayloadBytes = Math.max(
      200_000,
      Number(maxFramePayloadBytes) || DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
    );

    this.child = null;
    this.stdoutReader = null;
    this.stderrReader = null;
    this.ready = false;
    this.keepAlive = false;
    this.restartTimer = null;
    this.lastError = null;
    this.lastErrorAt = 0;
    this.lastLogLine = null;
    this.lastState = null;
    this.lastPrediction = null;
    this.lastWorkerMessageAt = 0;
    this.lastFrameForwardedAt = 0;
    this.lastReadyAt = 0;
    this.lastPongAt = 0;
    this.lastPingAt = 0;
    this.heartbeatTimer = null;
    this.frameInFlight = false;
    this.pendingFrameMeta = null;
    this.metrics = {
      framesReceived: 0,
      framesForwarded: 0,
      framesDroppedThrottle: 0,
      framesDroppedBusy: 0,
      framesDroppedOversize: 0,
      framesRejectedNotReady: 0,
      predictionsReceived: 0,
      statesReceived: 0,
      pingsSent: 0,
      pongsReceived: 0,
      lastLatencyMs: 0,
      avgLatencyMs: 0,
      latencySamples: 0,
    };
  }

  get running() {
    return Boolean(this.child);
  }

  getStatus() {
    return {
      running: this.running,
      ready: this.ready,
      pid: this.child?.pid ?? null,
      lastError: this.lastError,
      workerModule: this.workerModule,
      lastWorkerMessageAt: this.lastWorkerMessageAt || null,
      lastFrameForwardedAt: this.lastFrameForwardedAt || null,
      lastReadyAt: this.lastReadyAt || null,
      lastPongAt: this.lastPongAt || null,
      lastPingAt: this.lastPingAt || null,
      bufferSize: this.lastState?.buffer_size ?? 0,
      trackingOk: this.lastState?.tracking_ok ?? false,
      frameInFlight: this.frameInFlight,
      pendingFrameId: this.pendingFrameMeta?.clientFrameId ?? null,
      metrics: {
        ...this.metrics,
        avgLatencyMs: Number(this.metrics.avgLatencyMs.toFixed(2)),
      },
      lastLogLine: this.lastLogLine,
      lastState: this.lastState,
      lastPrediction: this.lastPrediction,
    };
  }

  async start() {
    this.keepAlive = true;
    if (this.child) {
      return this.getStatus();
    }

    this.lastError = null;
    this.lastLogLine = null;
    this.lastState = null;
    this.lastPrediction = null;
    this.lastWorkerMessageAt = 0;
    this.lastReadyAt = 0;
    const child = spawn(this.pythonBin, ["-m", this.workerModule], {
      cwd: this.projectRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
    });

    this.child = child;
    this.ready = false;
    this.frameInFlight = false;
    this.pendingFrameMeta = null;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    this.stdoutReader = readline.createInterface({ input: child.stdout });
    this.stderrReader = readline.createInterface({ input: child.stderr });
    console.info("[gesture-bridge] worker starting");

    this.stdoutReader.on("line", (line) => {
      this._handleStdoutLine(line);
    });

    this.stderrReader.on("line", (line) => {
      const message = String(line || "").trim();
      if (!message) {
        return;
      }
      this.lastLogLine = message;
      if (/traceback|error|exception|failed/i.test(message)) {
        this._setError(message);
        this.emit("message", { type: "error", message });
      } else {
        this.emit("message", { type: "log", level: "warn", message });
      }
      this.emit("status", this.getStatus());
    });

    child.on("error", (error) => {
      this._setError(error.message);
      this.emit("message", { type: "error", message: error.message });
      this.emit("status", this.getStatus());
    });

    child.on("exit", (code, signal) => {
      this._stopHeartbeat();
      this._teardownReaders();
      this.child = null;
      this.ready = false;
      this.frameInFlight = false;
      this.pendingFrameMeta = null;
      console.warn("[gesture-bridge] worker exited", { code, signal });
      this.emit("message", {
        type: "status",
        running: false,
        ready: false,
        exitCode: code,
        signal,
      });
      this.emit("status", this.getStatus());
      if (this.keepAlive) {
        this._scheduleRestart();
      }
    });

    this.emit("status", this.getStatus());
    return this.getStatus();
  }

  async stop() {
    this.keepAlive = false;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }

    if (!this.child) {
      this.ready = false;
      this._stopHeartbeat();
      return this.getStatus();
    }

    const child = this.child;
    this._sendMessage({ type: "shutdown" });
    await new Promise((resolve) => {
      const fallbackTimer = setTimeout(() => {
        if (this.child === child) {
          child.kill("SIGTERM");
        }
      }, 1200);

      child.once("exit", () => {
        clearTimeout(fallbackTimer);
        resolve();
      });
    });

    return this.getStatus();
  }

  async reset() {
    if (!this.child) {
      return this.getStatus();
    }
    this.lastState = null;
    this.lastPrediction = null;
    this.frameInFlight = false;
    this.pendingFrameMeta = null;
    this._sendMessage({ type: "reset" });
    return this.getStatus();
  }

  sendFrame({ ts, imageB64, clientFrameId, clientSentAtMs }) {
    this.metrics.framesReceived += 1;
    const normalizedClientFrameId = Number.isFinite(Number(clientFrameId))
      ? Number(clientFrameId)
      : Date.now();
    if (!this.child || !this.ready) {
      this.metrics.framesRejectedNotReady += 1;
      return { accepted: false, reason: "not_ready", clientFrameId: normalizedClientFrameId };
    }

    const payload = String(imageB64 || "");
    if (!payload) {
      return { accepted: false, reason: "empty_payload", clientFrameId: normalizedClientFrameId };
    }
    if (Buffer.byteLength(payload, "utf8") > this.maxFramePayloadBytes) {
      this.metrics.framesDroppedOversize += 1;
      this._setError("Frame payload is too large");
      this.emit("message", { type: "error", message: this.lastError });
      this.emit("status", this.getStatus());
      return { accepted: false, reason: "oversize", clientFrameId: normalizedClientFrameId };
    }

    const now = Date.now();
    if (now - this.lastFrameForwardedAt < this.minFrameIntervalMs) {
      this.metrics.framesDroppedThrottle += 1;
      return { accepted: false, reason: "throttle", clientFrameId: normalizedClientFrameId };
    }

    if (this.frameInFlight) {
      this.metrics.framesDroppedBusy += 1;
      return { accepted: false, reason: "busy", clientFrameId: normalizedClientFrameId };
    }

    this.lastFrameForwardedAt = now;
    this.frameInFlight = true;
    this.pendingFrameMeta = {
      clientFrameId: normalizedClientFrameId,
      sentAt: now,
      clientSentAtMs: Number.isFinite(Number(clientSentAtMs))
        ? Number(clientSentAtMs)
        : null,
    };
    this.metrics.framesForwarded += 1;
    this._sendMessage({
      type: "frame",
      ts: Number.isFinite(Number(ts)) ? Number(ts) : now,
      image_b64: payload,
      client_frame_id: this.pendingFrameMeta.clientFrameId,
      client_sent_at_ms: this.pendingFrameMeta.clientSentAtMs,
    });
    return { accepted: true, reason: null, clientFrameId: normalizedClientFrameId };
  }

  _handleStdoutLine(line) {
    const raw = String(line || "").trim();
    if (!raw) {
      return;
    }

    try {
      const message = normalizeGestureBridgeMessage(JSON.parse(raw));
      this.lastWorkerMessageAt = Date.now();
      if (message.type === "ready") {
        this.ready = true;
        this.lastReadyAt = Date.now();
        console.info("[gesture-bridge] worker ready");
        this._startHeartbeat();
      } else if (message.type === "pong") {
        this.lastPongAt = Date.now();
        this.metrics.pongsReceived += 1;
      } else if (message.type === "state") {
        this.lastState = message;
        this.metrics.statesReceived += 1;
        this._ackFrame(message);
      } else if (message.type === "prediction") {
        this.lastPrediction = message;
        this.metrics.predictionsReceived += 1;
        this._ackFrame(message);
      } else if (message.type === "error") {
        this._setError(message.message || "Unknown worker error");
      }

      this.emit("message", message);
      this.emit("status", this.getStatus());
    } catch (error) {
      this._setError(`Invalid worker JSON: ${error.message}`);
      this.emit("message", { type: "error", message: this.lastError });
      this.emit("status", this.getStatus());
    }
  }

  _sendMessage(message) {
    if (!this.child?.stdin?.writable) {
      return false;
    }
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
    return true;
  }

  _ackFrame(message) {
    const clientFrameId = Number(message.client_frame_id ?? 0);
    if (!this.pendingFrameMeta || clientFrameId !== this.pendingFrameMeta.clientFrameId) {
      return;
    }

    this.frameInFlight = false;
    const latencyMs = Date.now() - this.pendingFrameMeta.sentAt;
    this.metrics.lastLatencyMs = latencyMs;
    this.metrics.latencySamples += 1;
    this.metrics.avgLatencyMs +=
      (latencyMs - this.metrics.avgLatencyMs) / this.metrics.latencySamples;
    this.pendingFrameMeta = null;
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (!this.child || !this.ready) {
        return;
      }

      const now = Date.now();
      if (this.lastPingAt && now - this.lastPingAt > HEARTBEAT_TIMEOUT_MS && now - this.lastPongAt > HEARTBEAT_TIMEOUT_MS) {
        this._setError("Gesture worker heartbeat timed out");
        this.emit("message", { type: "error", message: this.lastError });
        this.emit("status", this.getStatus());
        this.child.kill("SIGTERM");
        return;
      }

      this.lastPingAt = now;
      this.metrics.pingsSent += 1;
      this._sendMessage({ type: "ping" });
    }, HEARTBEAT_INTERVAL_MS);
  }

  _stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    this.lastPingAt = 0;
    this.lastPongAt = 0;
  }

  _setError(message) {
    this.lastError = message;
    this.lastErrorAt = Date.now();
    console.error("[gesture-bridge]", message);
  }

  _scheduleRestart() {
    if (this.restartTimer || !this.keepAlive) {
      return;
    }
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      console.info("[gesture-bridge] restarting worker");
      this.start().catch((error) => {
        this._setError(error.message);
        this.emit("message", { type: "error", message: error.message });
        this.emit("status", this.getStatus());
      });
    }, 700);
  }

  _teardownReaders() {
    this.stdoutReader?.close();
    this.stderrReader?.close();
    this.stdoutReader = null;
    this.stderrReader = null;
  }
}


