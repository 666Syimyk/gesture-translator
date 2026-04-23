# Gesture Translator

## Live (GitHub Pages)

- App: https://666Syimyk.github.io/gesture-translator/?ui=user
- Repo: https://github.com/666Syimyk/gesture-translator

Примечание: GitHub Pages публикует только `frontend` (статический сайт). Если интерфейс пытается обратиться к `backend` (эндпоинты `/api`), эти функции будут работать только при запущенном backend’е (локально или на отдельном хостинге).

## Установка на телефон (PWA)

GitHub Pages устанавливается как PWA (веб‑приложение), а не как APK.

- Android (Chrome): меню `⋮` → **Install app / Установить приложение**
- iPhone (Safari): **Поделиться** → **На экран «Домой»**

Если видите “Request failed” или часть функций не работает — нужен доступный backend (см. ниже).

## Backend URL для GitHub Pages

На GitHub Pages backend не запускается. Чтобы функции, которые ходят в `/api`, работали на телефоне, backend нужно запустить локально или задеплоить на отдельный хостинг с HTTPS.

В интерфейсе можно указать адрес backend’а: **Settings → Backend → API base URL → Save & reload**.

## Структура

```text
gesture-translator/
  frontend/   React + Vite интерфейс
  backend/    Express API, PostgreSQL, ML-скрипты и датасеты
  docs/       документация по сбору данных и обучению
```

## Запуск (локально)

Из корня проекта в двух окнах PowerShell:

```powershell
npm run dev:backend
```

```powershell
npm run dev:frontend
```

Адреса:

- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:5000/api/health`

## Режимы интерфейса

- Принудительно открыть `User mode`: добавьте `?ui=user` (например, `http://localhost:5173/?ui=user`)
- Принудительно открыть `Admin mode`: добавьте `?ui=admin`

## GitHub Pages (деплой)

Деплой фронтенда на GitHub Pages настроен через GitHub Actions: `.github/workflows/deploy-pages.yml`.

Если GitHub Actions не запускаются (например, из‑за ограничений аккаунта), можно задеплоить вручную в ветку `gh-pages`:

```powershell
.\scripts\deploy-gh-pages.ps1
```

После этого в GitHub откройте: `Settings → Pages → Build and deployment → Deploy from a branch` и выберите `gh-pages` + `/(root)`.
