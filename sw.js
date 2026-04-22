const CACHE_NAME = "silent-conversation-v2";

function getBasePath() {
  try {
    const scope = String(self.registration?.scope ?? "");
    const pathname = new URL(scope).pathname;
    return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  } catch {
    return "";
  }
}

const BASE_PATH = getBasePath();
const withBase = (path) => `${BASE_PATH}${path}`;
const APP_SHELL = [
  withBase("/"),
  withBase("/?ui=user"),
  withBase("/manifest.webmanifest"),
  withBase("/favicon.svg"),
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).catch(() => {}),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key)),
      ),
    ),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((networkResponse) => {
          const responseClone = networkResponse.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(withBase("/"), responseClone))
            .catch(() => {});
          return networkResponse;
        })
        .catch(() => caches.match(withBase("/")) || Response.error()),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(request)
        .then((networkResponse) => {
          if (
            request.destination === "document" ||
            request.headers.get("accept")?.includes("text/html")
          ) {
            return networkResponse;
          }

          const responseClone = networkResponse.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(request, responseClone))
            .catch(() => {});
          return networkResponse;
        })
        .catch(() => caches.match(withBase("/")) || Response.error());
    }),
  );
});
