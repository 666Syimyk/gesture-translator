# Data Collection Workflow

## Цель

Подготовить чистый `approved + ready` датасет для:

- `alphabet`
- `sign`
- `phrase`

и затем запустить:

- dataset export
- benchmark
- выбор лучшей модели

## 1. Подготовить словарь с экспертом

Заполните JSON-шаблоны:

- `docs/templates/letters.rsl.template.json`
- `docs/templates/signs.rsl.template.json`
- `docs/templates/phrases.rsl.template.json`

Для каждой записи согласуйте:

- `text`
- `recognitionLevel`
- `unitCode`
- `description`
- `referenceNotes`

## 2. Импортировать словарь в базу

Из папки `server/`:

```bash
npm run import:labels -- --file ..\\docs\\templates\\letters.rsl.template.json
npm run import:labels -- --file ..\\docs\\templates\\signs.rsl.template.json
npm run import:labels -- --file ..\\docs\\templates\\phrases.rsl.template.json
```

Если нужен другой язык:

```bash
npm run import:labels -- --file ..\\docs\\templates\\phrases.rsl.template.json --sign-language rsl
```

## 3. Записывать примеры

На каждого исполнителя:

- используйте один и тот же язык версии 1
- записывайте буквы, знаки и фразы отдельно
- держите лицо, руки и корпус в кадре
- не подтверждайте плохие или слишком быстрые записи

Минимальный практический старт:

- 5 исполнителей
- 15-20 повторов на класс
- сначала закрыть первую волну фраз и базовые знаки

## 4. Делать review

Каждый ролик должен получить:

- `approved` или `rejected`
- `quality_score`
- `review_notes`

В обучение идут только:

- `approved`
- `landmarks ready`
- quality выше порога

## 5. Проверить extraction quality

Смотрите на:

- `valid_frame_ratio`
- `missing_hand_ratio`
- `missing_face_ratio`
- `missing_pose_ratio`

Если sequence слабый, не пускайте его в датасет.

## 6. Сделать dataset export

Экспорт строится отдельно по scope:

- `alphabet`
- `sign`
- `phrase`
- `unified`

Нормальная последовательность:

1. сначала собрать и проверить данные
2. потом сделать export
3. потом запускать benchmark

## 7. Запустить benchmark

Когда данных достаточно:

- сравните `baseline`
- сравните `gru`
- сравните `tcn`

Смотрите на:

- `top-1`
- `top-3`
- `per-class accuracy`
- `confusion pairs`
- `weak phrases`
- `latency`
- `low-confidence rate`

## 8. Что делать дальше

После benchmark:

1. выбрать победителя
2. включить его как активную live-модель
3. добирать данные по слабым классам
4. повторить benchmark
