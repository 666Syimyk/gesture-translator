# Phrase Sequence Recognition

Этот контур нужен для закрытого распознавания коротких фраз по движению во времени. Он не делает полноценный continuous sign language translation; первая версия решает более реалистичную задачу: один клип 1-3 секунды -> одна фраза из ограниченного словаря.

## Главный Конфиг

Список фраз, пути, качество записи, augmentation и live-настройки лежат в одном месте:

```text
backend/ml/phrase_sequence/default_config.json
```

По умолчанию включены классы:

- `Дом`
- `hello`
- `thanks`
- `yes`
- `no`
- `help`
- `water`
- `repeat`
- `stop`
- `understand`
- `dont_understand`
- `none`

`none` обязателен: это состояние покоя, чтобы live-режим не угадывал фразы постоянно.

## Датасет

Структура:

```text
backend/uploads/datasets/phrase_sequence/latest/
  train/
    hello/
    none/
  val/
    hello/
    none/
  test/
    hello/
    none/
```

Внутри классов сохраняются JSON-файлы с landmarks. Запись использует MediaPipe Hand + Pose + Face Landmarker.

## Установка

Из корня проекта:

```powershell
pip install -r backend/requirements.txt
```

## Запись Данных

Одна метка:

```powershell
npm run phrase:record -- --label hello --split train --samples 50
```

Список меток:

```powershell
npm run phrase:record -- --labels hello,thanks,yes --split train --samples 50
```

Все метки из конфига:

```powershell
npm run phrase:record -- --all-labels --split val --samples 10
```

Готовые preset-команды для полного старта:

```powershell
npm run phrase:record:train
npm run phrase:record:val
npm run phrase:record:test
```

Отдельно для жеста `Привет` используется label `hello`:

```powershell
npm run phrase:record:privet:train
npm run phrase:record:privet:val
npm run phrase:record:privet:test
```

Для фразы `Дом` добавлены отдельные быстрые команды:

```powershell
npm --prefix backend run phrase:record:dom:train
npm --prefix backend run phrase:record:dom:val
npm --prefix backend run phrase:record:dom:test
```

Класс покоя:

```powershell
npm run phrase:record -- --label none --split train --samples 50
```

Плохие записи не сохраняются: для обычных фраз проверяется наличие landmarks, а для `none` проверяется только то, что клип не пустой.

## Summary Датасета

```powershell
npm run phrase:summary
```

Команда показывает `usable/total` по каждому `label` и `split`, чтобы быстро увидеть, каких данных не хватает.

## Обучение

```powershell
npm run phrase:train
```

По умолчанию используется LSTM, окно `32` кадра, early stopping, feature mode `full` и простая landmark augmentation на train-сэмплах.

## Оценка

```powershell
npm run phrase:evaluate
```

Результаты:

```text
backend/ml/artifacts/phrase_sequence/latest/evaluation.json
```

Там есть accuracy, confusion matrix и classification report.

## Экспорт Лучшей Модели

```powershell
npm run phrase:export -- --overwrite
```

Экспорт копирует текущую обученную модель в:

```text
backend/ml/artifacts/phrase_sequence/best/
```

## Live Inference

```powershell
npm run phrase:live
```

Live-режим использует скользящее окно кадров, smoothing, stable votes, confidence threshold, cooldown между одинаковыми фразами, лог последних предсказаний и класс `none`. Если уверенность ниже порога или модель видит `none`, фраза не озвучивается.

## Хороший Старт

Для первой нормальной проверки запиши:

- `train`: 50-100 примеров на класс
- `val`: 10-20 примеров на класс
- `test`: 10-20 примеров на класс

Снимай при разном свете и обязательно запиши много `none`.
