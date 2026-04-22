# Gesture Translator

Веб-приложение для распознавания жестов и коротких фраз с камерой, backend API и ML-пайплайном.

## Структура

```text
gesture-translator/
  frontend/   React + Vite интерфейс
  backend/    Express API, PostgreSQL, ML-скрипты и датасеты
  docs/       документация по сбору данных и обучению
```

## Запуск приложения

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

- На телефоне по умолчанию открывается `User mode` (мобильный интерфейс).
- Принудительно открыть `Admin mode`: добавьте `?ui=admin` (например, `http://localhost:5173/?ui=admin`).
- На десктопе по умолчанию открывается `Admin mode`. Открыть `User mode`: `?ui=user` или путь `/user`.

## Установка backend

```powershell
cd backend
npm install
pip install -r requirements.txt
```

Создайте файл `backend/.env` на основе `backend/.env.example` и заполните параметры подключения к PostgreSQL.

## Новый контур фраз по движению

Для коротких фраз теперь есть отдельный closed-set sequence-пайплайн:

- MediaPipe Hand + Pose + Face Landmarker
- окно последовательности `20-40` кадров, по умолчанию `32`
- классы `hello`, `thanks`, `yes`, `no`, `help`, `water`, `repeat`, `stop`, `understand`, `dont_understand`, `none`
- `none` нужен, чтобы модель не угадывала фразы в состоянии покоя
- LSTM-модель с early stopping
- отдельные команды записи датасета, обучения, оценки и live inference

Подробно: [docs/PHRASE_SEQUENCE.md](docs/PHRASE_SEQUENCE.md)

Быстрые команды из папки `backend`:

```powershell
npm run phrase:record -- --label hello --split train --samples 10
npm run phrase:record -- --labels hello,thanks,yes --split train --samples 10
npm run phrase:record -- --all-labels --split val --samples 5
npm run phrase:record:train
npm run phrase:record:val
npm run phrase:record:test
npm run phrase:record:privet:train
npm run phrase:record:privet:val
npm run phrase:record:privet:test
npm run phrase:summary
npm run phrase:train
npm run phrase:evaluate
npm run phrase:export -- --overwrite
npm run phrase:live
```

## Важно

Это не полноценный continuous sign language translation. Первый рабочий этап проекта - распознавание одного короткого клипа как одной фразы из ограниченного словаря. Для качества в реальной камере нужно записать свои примеры каждого класса, особенно `none`.
