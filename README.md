# Gesture Translator

## Live (GitHub Pages)

- App: https://666Syimyk.github.io/gesture-translator/?ui=user
- Repo: https://github.com/666Syimyk/gesture-translator

Примечание: GitHub Pages публикует только `frontend` (статический сайт). Если интерфейс пытается обратиться к `backend` (эндпоинты `/api`), эти функции будут работать только при запущенном backend’е (локально или на отдельном хостинге).

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

