import { request } from "./apiClient";

function formatHistoryItem(item) {
  const createdAt = item.created_at ?? item.createdAt ?? new Date().toISOString();

  return {
    id: item.id,
    type: item.type,
    text: item.text,
    createdAt,
    time: new Date(createdAt).toLocaleTimeString("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
    }),
  };
}

export async function fetchHistory() {
  const items = await request("/api/history");
  return items.map(formatHistoryItem);
}

export async function createHistoryItem(payload) {
  const item = await request("/api/history", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  return formatHistoryItem(item);
}

export async function clearHistoryRequest() {
  return request("/api/history", {
    method: "DELETE",
  });
}
