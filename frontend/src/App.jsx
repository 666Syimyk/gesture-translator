import { useEffect, useRef, useState } from "react";
import "./App.css";
import CameraView from "./components/CameraView";
import { fetchBootstrapData } from "./api/bootstrapApi";
import { fetchLatestModel, predictLatestModel } from "./api/mlApi";
import { updateSettings } from "./api/settingsApi";
import { useGestureBridge } from "./hooks/useGestureBridge";
import { extractHandFeatures, predictGesture } from "./gestureUtils";
import { speakWithSettings } from "./utils/speech";
import { resolveGestureLabelRu } from "./utils/gestureLabels";

const copy = {
  ru: {
    htmlLang: "ru",
    speechLocale: "ru-RU",
    nav: { home: "Главная", camera: "Камера", voice: "Голос", settings: "Настройки" },
    home: {
      badge: "Silent conversation",
      title: "Silent conversation",
      subtitle: "Жесты в текст и голос — в реальном времени",
      text: "Показывайте жесты в камеру — приложение распознаёт их по кадрам и выводит понятный текст. При желании результат можно озвучивать автоматически.",
      features: [
        [
          "Распознавание по кадрам",
          "Система отслеживает руки и движение, чтобы уверенно распознавать жесты в реальном времени.",
        ],
        [
          "Слова и фразы",
          "Собирайте слова из букв или показывайте жесты — распознанный результат появляется сразу на экране.",
        ],
        [
          "Озвучка и настройки",
          "Включите «Авто-озвучку», настройте тему и язык — приложение подстроится под вас.",
        ],
      ],
    },
    camera: {
      badge: "Камера",
      title: "Камера жестов",
      text: "Покажите букву или жест «Привет» в камеру, а приложение распознает их в реальном времени.",
      phraseLibraryBadge: "Фразы",
      phraseLibraryTitle: "Все доступные фразы",
      phraseLibraryText: "Нажмите на фразу, чтобы открыть окно с инструкцией показа.",
      phraseGuideBadge: "Инструкция",
      phraseGuideOpen: "Как показать",
      phraseGuideClose: "Закрыть",
      phraseGuideLiveTitle: "Проверка в кадре",
      faceZoneHint: "Область лица",
      alphabet: "Буквы",
      sign: "Жесты",
      phrase: "Фразы",
      lettersTitle: "Буквы",
      wordTitle: "Слово",
      lettersPlaceholder: "Здесь появятся распознанные буквы",
      wordPlaceholder: "Покажите буквы, чтобы собрать слово",
      wordSpace: "Пробел",
      wordSpeak: "Озвучить",
      wordBackspace: "Удалить",
      wordClear: "Очистить",
      start: "Запустить камеру",
      stop: "Отключить камеру",
      front: "Передняя камера",
      rear: "Задняя камера",
      idle: "Нажмите кнопку, чтобы включить камеру.",
      waiting: "Ждём жест...",
      waitingAlphabet: "Ждём букву...",
      analyzing: "Анализируем...",
      guideTitle: "Как показать «Привет»",
      guideText: "Подсказки помогут камере точнее распознать приветственный жест.",
      guidePokaHint: "Для «Пока»: покажите одну руку и 1–2 раза откройте/закройте ладонь.",
      guidePoseHint: "Держите верхнюю часть корпуса в кадре, чтобы камера видела плечи.",
      guideVisionTitle: "ИИ видит сейчас",
      guideItems: {
        oneHand: "Покажите одну руку",
        wave: "Махните 1–2 раза",
        shoulder: "Держите руку выше плеча",
      },
      guideVisionItems: {
        head: "Голова",
        face: "Лицо",
        eyes: "Глаза",
        nose: "Нос",
        lips: "Губы",
        ears: "Уши",
        neck: "Шея",
        shoulder: "Плечо",
        elbow: "Локоть",
        hand: "Кисть",
        fingers: "Пальцы",
      },
      phraseLibraryItems: {
        da: "Да",
        net: "Нет",
        privet: "Привет",
        poka: "Пока",
        dom: "Дом",
        druzhba: "Дружба",
        muzhchina: "Мужчина",
        zhenshchina: "Женщина",
        solntse: "Солнце",
        ya: "Я",
        ty: "Ты",
        horosho: "Хорошо",
        stop: "Стоп",
        est: "Есть",
        pit: "Пить",
        spat: "Спать",
        idti: "Идти",
        bezhat: "Бежать",
        dumat: "Думать",
        lyubov: "Любовь",
        bolshoy: "Большой",
        malenkiy: "Маленький",
        krasiviy: "Красивый",
        spasibo: "Спасибо",
      },
      phraseGuideItems: {
        da: {
          title: "Как показать «Да»",
          text: "Кулак перед грудью, 2-3 коротких движения вверх-вниз, как маленький кивок.",
          steps: [
            "Сожмите правую руку в плотный кулак, большой палец сверху, не выставляйте его в сторону.",
            "Держите кулак перед верхней частью груди, локоть слегка согните и не отводите далеко от корпуса.",
            "Сделайте 2-3 коротких компактных движения вверх-вниз, не разжимая кулак и без широкого замаха.",
          ],
        },
        net: {
          title: "Как показать «Нет»",
          text: "Указательный и средний выпрямлены рядом, затем быстро смыкаются.",
          steps: [
            "Поднимите правую руку перед грудью и держите кисть сбоку, не слишком далеко от тела.",
            "Согните безымянный, мизинец и большой палец, а указательный и средний выпрямите вместе и рядом.",
            "Сделайте короткое аккуратное движение к позиции жеста и быстро сомкните два пальца 1-2 раза без широкой амплитуды.",
          ],
        },
        privet: {
          title: "Как показать «Привет»",
          text: "Одна открытая ладонь в верхней зоне рядом с головой делает короткий дружелюбный мах.",
          steps: [
            "Поднимите правую руку к зоне головы или плеча, рядом с лицом.",
            "Откройте ладонь и держите кисть в верхней зоне кадра.",
            "Сделайте короткий дружелюбный мах или лёгкий наклон ладони без большой амплитуды.",
          ],
        },
        poka: {
          title: "Как показать «Пока»",
          text: "Открытая ладонь у головы делает повторяемые колебания из стороны в сторону.",
          steps: [
            "Поднимите открытую правую ладонь к зоне головы.",
            "Сделайте 2-4 маленьких маха из стороны в сторону.",
            "Можно использовать разговорный вариант: несколько открытий и закрытий ладони, как «пока-пока».",
          ],
        },
        dom: {
          title: "Как показать «Дом»",
          text: "Обе руки перед грудью образуют чёткую крышу и треугольник.",
          steps: [
            "Поднимите обе руки перед грудью, ладони смотрят друг на друга.",
            "Соедините кончики пальцев и соберите аккуратный треугольник-домик.",
            "Удержите форму 1-2 секунды, не разводя кисти в стороны и не опуская руки.",
          ],
        },
        druzhba: {
          title: "Как показать «Дружба»",
          text: "Пальцы обеих рук переплетаются в замок, показывая крепкую связь и единство.",
          steps: [
            "Поднимите обе руки перед грудью.",
            "Переплетите пальцы обеих рук в замок, сцепив их крепко между собой.",
            "Ненадолго удержите замок, чтобы жест был читаемым и устойчивым.",
          ],
        },
        muzhchina: {
          title: "Как показать «Мужчина»",
          text: "Короткий маркирующий жест в верхней зоне лица: у виска, лба или линии «козырька».",
          steps: [
            "Поднимите правую руку к верхней зоне лица, ближе к виску или лбу.",
            "Коснитесь этой точки или подведите руку очень близко.",
            "Сделайте короткое маркирующее движение без большой амплитуды и не опускайте жест слишком низко.",
          ],
        },
        zhenshchina: {
          title: "Как показать «Женщина»",
          text: "Открытая ладонь боком: коснитесь правой щеки, затем левой щеки.",
          steps: [
            "Откройте ладонь, пальцы держите вместе, ладонь поверните боком (ребром к камере).",
            "Коснитесь (или подведите очень близко) правой щеки.",
            "Перенесите ладонь на левую щеку и снова коснитесь. Движение спокойное, без резких махов.",
            "Важно: если ладонь смотрит прямо в камеру, распознавание будет хуже — держите её боком.",
          ],
        },
        solntse: {
          title: "Как показать «Солнце»",
          text: "Открытая ладонь выше головы и компактное круговое движение кистью.",
          steps: [
            "Поднимите правую ладонь выше макушки или чуть сбоку сверху.",
            "Пальцы держите прямыми, вместе или слегка раскрытыми.",
            "Сделайте компактный круг кистью, как будто обозначаете солнце.",
          ],
        },
        ya: {
          title: "Как показать «Я»",
          text: "Указательный палец направлен на себя, прямо в грудь.",
          steps: [
            "Поднимите правую руку перед грудью и выпрямите указательный палец.",
            "Остальные пальцы согните и направьте палец именно в центр своей груди.",
            "Коротко зафиксируйте жест, не уводя руку далеко в сторону и не направляя палец вперёд.",
          ],
        },
        ty: {
          title: "Как показать «Ты»",
          text: "Указательный палец направлен прямо вперёд, на человека перед вами.",
          steps: [
            "Выпрямите указательный палец правой рукой, остальные пальцы согните.",
            "Держите руку перед грудью и направьте палец вперёд от себя, а не в грудь.",
            "Коротко зафиксируйте направление, чтобы жест был читаемым и устойчивым.",
          ],
        },
        horosho: {
          title: "Как показать «Хорошо»",
          text: "Кулак и большой палец строго вверх.",
          steps: [
            "Сожмите правую руку в кулак.",
            "Поднимите большой палец строго вверх, не разворачивая кисть боком.",
            "Удержите жест стабильно перед камерой.",
          ],
        },
        stop: {
          title: "Как показать «Стоп»",
          text: "Открытая ладонь направлена на экран, как чёткий знак остановки.",
          steps: [
            "Поднимите правую руку перед собой.",
            "Раскройте ладонь полностью, пальцы держите вместе и ровно.",
            "Удержите положение, не раскачивая кисть и не делая махов.",
          ],
        },
        est: {
          title: "Как показать «Есть»",
          text: "Пальцы собираются в щепоть и 1-2 раза подносятся к губам, как будто вы едите.",
          steps: [
            "Соедините пальцы правой руки в щепоть, как будто держите щепотку соли.",
            "Поднесите щепоть к губам и слегка согните кисть к рту.",
            "Повторите движение 1-2 раза, не смешивая жест с «Спасибо» или «Мужчина».",
          ],
        },
        pit: {
          title: "Как показать «Пить»",
          text: "Рука имитирует стакан и подносится к нижней губе 1-2 раза, как при питье.",
          steps: [
            "Сделайте кистью форму, как будто держите чашку или стакан.",
            "Поднесите этот «стакан» к нижней губе, будто делаете глоток.",
            "Повторите движение 1-2 раза, без резких рывков и лишних махов.",
          ],
        },
        spat: {
          title: "Как показать «Спать»",
          text: "Ладонь опускается перед лицом к подбородку, а внизу пальцы мягко смыкаются, как во сне.",
          steps: [
            "Расположите руку перед лицом, ладонью к себе, пальцы слегка расставлены.",
            "Плавно опустите руку вниз к подбородку.",
            "Когда рука дойдёт до подбородка, мягко сомкните пальцы и закройте глаза.",
          ],
        },
        idti: {
          title: "Как показать «Идти»",
          text: "Два пальца «шагают» по воздуху в спокойном темпе.",
          steps: [
            "Покажите указательный и средний пальцы как «ножки».",
            "Сделайте поочерёдные шаги пальцами.",
            "Держите ровный спокойный темп, без ускорения.",
          ],
        },
        bezhat: {
          title: "Как показать «Бежать»",
          text: "Тот же «шаг» двумя пальцами, но быстрее и активнее.",
          steps: [
            "Сохраните форму двух пальцев как для «Идти».",
            "Сделайте частые и быстрые поочерёдные шаги.",
            "Добавьте энергичный темп, но без хаотичных взмахов рукой.",
          ],
        },
        dumat: {
          title: "Как показать «Думать»",
          text: "Указательный палец к виску, как знак мысли.",
          steps: [
            "Выпрямите указательный палец, остальные пальцы согните.",
            "Поднесите палец к виску и коснитесь или подведите очень близко.",
            "Удержите короткую фиксацию, без широких движений.",
          ],
        },
        lyubov: {
          title: "Как показать «Любовь»",
          text: "Обе руки скрещены на груди в мягком самообъятии.",
          steps: [
            "Поднимите обе руки к верхней части груди.",
            "Скрестите руки как мягкое самообъятие, ладони на плечах или верхней груди.",
            "Локти опустите вниз и чуть в стороны, удержите 1-2 секунды без спешки.",
          ],
        },
        bolshoy: {
          title: "Как показать «Большой»",
          text: "Две руки на уровне груди расходятся в стороны, показывая большую ширину.",
          steps: [
            "Поднимите обе руки на уровень груди, ладони смотрят друг на друга.",
            "Начните с меньшей дистанции и разведите руки в стороны.",
            "Покажите один чёткий большой размер без дёргания и без лишних повторов.",
          ],
        },
        malenkiy: {
          title: "Как показать «Маленький»",
          text: "Две руки или пальцы показывают маленькую дистанцию и маленький размер.",
          steps: [
            "Держите руки на уровне груди и оставьте между ними маленький зазор.",
            "Покажите малую ширину без широкого разведения.",
            "Можно использовать щепоть или очень близкие пальцы как компактный вариант.",
          ],
        },
        krasiviy: {
          title: "Как показать «Красивый»",
          text: "Одна рука в лицевой зоне делает короткое плавное движение вдоль щеки или от лица наружу.",
          steps: [
            "Поднесите одну руку к щеке или общей зоне лица.",
            "Сделайте короткое плавное движение вдоль щеки или мягко от лица наружу.",
            "Не превращайте жест в простое касание, как у «Женщина», и не стартуйте от подбородка как у «Спасибо».",
          ],
        },
        spasibo: {
          title: "Как показать «Спасибо»",
          text: "Кулак касается только лба. Покажите касание по кадрам: коснулись → удержали → убрали.",
          steps: [
            "Сожмите правую руку в кулак, большой палец сверху.",
            "Поднимите кулак к лбу и коснитесь (или подведите очень близко).",
            "Удержите касание 0.3–0.8 сек — это важный кадр для распознавания.",
            "Уберите руку спокойно, без махов в стороны и без повторов.",
          ],
        },
      },
      phraseChecklistItems: {
        da: [
          "Кулак собран, большой палец сверху",
          "Позиция перед верхней частью груди",
          "2-3 коротких движения вверх-вниз",
        ],
        net: [
          "Указательный и средний рядом",
          "Кисть слегка боком, не ладонью в лицо",
          "Быстрое смыкание 1-2 раза без широкой амплитуды",
        ],
        privet: ["Открытая ладонь", "Зона головы/плеча", "Короткий дружелюбный мах"],
        poka: ["Открытая ладонь", "Зона головы", "2-4 колебания или «пока-пока»"],
        dom: ["Обе руки", "Треугольник-«крыша»", "Форма удерживается"],
        druzhba: ["Обе руки", "Пальцы в замок", "Крепкое сцепление"],
        muzhchina: ["Верхняя зона лица", "Точка у виска или лба", "Короткий маркирующий жест"],
        zhenshchina: ["Нижняя зона лица", "Щека или уголок губ", "1-2 коротких касания"],
        solntse: ["Ладонь выше головы", "Компактный круг кистью", "Пальцы как лучи"],
        ya: ["Указательный палец", "На грудь", "Я сам/сама"],
        ty: ["Указательный палец", "Вперёд", "На другого человека"],
        horosho: ["Кулак", "Большой палец вверх", "Кисть не боком"],
        stop: ["Открытая ладонь вперёд", "Без махов", "Жест как стоп-сигнал"],
        est: ["Три пальца вместе", "Ко рту", "Как будто ешь"],
        pit: ["Рука как стакан", "К нижней губе", "1-2 глотка"],
        spat: ["Ладонь перед лицом", "Плавно вниз к подбородку", "Закрыть глаза и сомкнуть пальцы"],
        idti: ["Два пальца как ножки", "Спокойный шаг", "Поочерёдно"],
        bezhat: ["Два пальца как ножки", "Быстрее и активнее", "Ускоренный шаг"],
        dumat: ["Указательный палец", "К виску", "Мысль/размышление"],
        lyubov: ["Обе руки на груди", "Мягкое самообъятие", "Поза удерживается"],
        bolshoy: ["Две руки на груди", "Разведение в стороны", "Один чёткий большой размер"],
        malenkiy: ["Две руки или пальцы", "Малая дистанция", "Компактная амплитуда"],
        krasiviy: ["Лицевая зона", "Короткое плавное движение", "Не от подбородка"],
        spasibo: ["Кулак", "Касание лба", "Короткая фиксация"],
      },
    },
    voice: {
      badge: "Распознавание голоса",
      title: "Говорите — приложение распознает",
      text: "Запишите голос, и Silent conversation преобразует речь в текст для быстрого и удобного общения.",
      start: "Начать запись",
      stop: "Остановить запись",
      recording: "Идёт запись",
      transcript: "Текст записи",
      transcriptEmpty: "Здесь появится текст после записи голоса.",
      transcriptHint: "После распознавания вы сможете прочитать результат и использовать его дальше.",
      micLabel: "Кнопка микрофона",
    },
    settings: {
      badge: "Настройки",
      title: "Тема и язык",
      text: "Здесь можно настроить внешний вид приложения и язык интерфейса.",
      uiModeTitle: "Режим интерфейса",
      uiModeText: "Переключайтесь между Admin и User режимами.",
      autoSpeakTitle: "Авто-озвучка",
      autoSpeakText: "После распознавания приложение будет озвучивать слова вслух.",
      themeTitle: "Тема",
      themeText: "Выберите светлую или тёмную тему.",
      languageTitle: "Язык",
      languageText: "Выберите язык интерфейса.",
      previewTitle: "Сейчас активно",
      previewText: "Изменения применяются сразу ко всему интерфейсу.",
    },
    options: {
      theme: { dark: "Тёмная", light: "Светлая" },
      language: { ru: "Русский", en: "English" },
      autoSpeak: { on: "Вкл", off: "Выкл" },
      uiMode: { admin: "Admin", user: "User" },
    },
    speech: {
      ready: "Нажмите кнопку и начните говорить.",
      unsupported: "В этом браузере распознавание речи недоступно. Лучше открыть приложение в Chrome или Edge.",
      listening: "Слушаю вас. Говорите в микрофон.",
      transforming: "Преобразую речь в текст.",
      finishedSaved: "Запись завершена. Текст сохранён.",
      stopped: "Запись остановлена.",
      stopping: "Останавливаю запись...",
      languageChanged: "Язык переключён. Теперь можно записывать на новом языке.",
      startError: "Не удалось запустить запись. Проверьте доступ к микрофону.",
      errors: {
        "audio-capture": "Микрофон не найден. Проверьте устройство записи.",
        network: "Для распознавания речи нужно подключение к сети.",
        "no-speech": "Речь не распознана. Попробуйте ещё раз.",
        "not-allowed": "Разрешите доступ к микрофону в браузере.",
        "service-not-allowed": "Браузер запретил сервис распознавания речи.",
        unknown: "Не удалось распознать речь. Попробуйте ещё раз.",
      },
    },
  },
  en: {
    htmlLang: "en",
    speechLocale: "en-US",
    nav: { home: "Home", camera: "Camera", voice: "Voice", settings: "Settings" },
    home: {
      badge: "Gesture Translator",
      title: "Gesture Translator",
      subtitle: "Understanding each other can be easier than it seems",
      text: "The app recognizes gestures in real time, turns them into text, and makes communication easier.",
      features: [
        ["Gestures, letters, and phrases", "The system analyzes hand movement and recognizes letters, signs, and short phrases."],
        ["Instant result on screen", "Recognized text appears immediately so users can quickly see the result."],
        ["Voice and smart settings", "Speech output and simple interface settings are built in."],
      ],
    },
    camera: {
      badge: "Camera",
      title: "Camera",
      text: "Show a letter or the “Hello” gesture for live recognition.",
      phraseLibraryBadge: "Phrases",
      phraseLibraryTitle: "All available phrases",
      phraseLibraryText: "Tap a phrase to open a guide window with showing steps.",
      phraseGuideBadge: "Guide",
      phraseGuideOpen: "How to show",
      phraseGuideClose: "Close",
      phraseGuideLiveTitle: "Live checklist",
      faceZoneHint: "Face area",
      alphabet: "Letters",
      sign: "Gestures",
      phrase: "Phrases",
      lettersTitle: "Letters",
      wordTitle: "Word",
      lettersPlaceholder: "Recognized letters will appear here",
      wordPlaceholder: "Show letters to build a word",
      wordSpace: "Space",
      wordSpeak: "Speak",
      wordBackspace: "Backspace",
      wordClear: "Clear",
      start: "Launch camera",
      stop: "Turn off camera",
      front: "Front camera",
      rear: "Rear camera",
      idle: "Tap the button to turn on the camera.",
      waiting: "Waiting for a gesture...",
      waitingAlphabet: "Waiting for a letter...",
      analyzing: "Analyzing...",
      guideTitle: "How to show “Hello”",
      guideText: "These tips help the camera recognize a greeting wave more reliably.",
      guidePokaHint: "For “Bye”: show one hand and open/close your palm 1-2 times.",
      guidePoseHint: "Keep your upper body in frame so the camera can see your shoulders.",
      guideVisionTitle: "AI can see now",
      guideItems: {
        oneHand: "Show one hand",
        wave: "Wave 1-2 times",
        shoulder: "Keep your hand above the shoulder",
      },
      guideVisionItems: {
        head: "Head",
        face: "Face",
        eyes: "Eyes",
        nose: "Nose",
        lips: "Lips",
        ears: "Ears",
        neck: "Neck",
        shoulder: "Shoulder",
        elbow: "Elbow",
        hand: "Hand",
        fingers: "Fingers",
      },
      phraseLibraryItems: {
        da: "Yes",
        net: "No",
        privet: "Hello",
        poka: "Bye",
        dom: "Home",
        druzhba: "Friendship",
        muzhchina: "Man",
        zhenshchina: "Woman",
        solntse: "Sun",
        ya: "I",
        ty: "You",
        horosho: "Good",
        stop: "Stop",
        est: "Eat",
        pit: "Drink",
        spat: "Sleep",
        idti: "Walk",
        bezhat: "Run",
        dumat: "Think",
        lyubov: "Love",
        bolshoy: "Big",
        malenkiy: "Small",
        krasiviy: "Beautiful",
        spasibo: "Thank you",
      },
      phraseGuideItems: {
        da: {
          title: "How to show “Yes”",
          text: "A fist moving up-down like a confirming nod.",
          steps: [
            "Close one hand into a fist.",
            "Keep the fist in front of your chest.",
            "Move the fist up-down 1-2 times.",
          ],
        },
        net: {
          title: "How to show “No”",
          text: "Extend index and middle, then quickly close fingers.",
          steps: [
            "Extend index and middle fingers.",
            "Keep the thumb under them.",
            "Quickly close fingers 1-2 times.",
          ],
        },
        privet: {
          title: "How to show “Hello”",
          text: "Raise an open palm above the shoulder and move it lightly side to side like a greeting.",
          steps: [
            "Raise one hand above shoulder level so the camera can clearly see it.",
            "Keep the palm open with fingers straight or slightly relaxed.",
            "Make 1-2 light side-to-side greeting motions.",
          ],
        },
        poka: {
          title: "How to show “Bye”",
          text: "Classic wave: open palm with wrist turns.",
          steps: [
            "Raise one open palm.",
            "Keep the palm in front of camera.",
            "Turn wrist left-right several times.",
          ],
        },
        dom: {
          title: "How to show “Home”",
          text: "A roof shape made with both hands.",
          steps: [
            "Raise both hands in front of you.",
            "Connect fingertips to form a triangle.",
            "Hold the roof shape steady for a second.",
          ],
        },
        druzhba: {
          title: "How to show “Friendship”",
          text: "Meaning is connection: interlocked fists or crossed index fingers.",
          steps: [
            "Make fists with both hands.",
            "Interlock fists at chest level and hold.",
            "Variant 2: cross index fingers and hold.",
          ],
        },
        muzhchina: {
          title: "How to show “Man”",
          text: "Forehead zone is key: touch and slight forward move.",
          steps: [
            "Use open hand or index finger.",
            "Touch the forehead.",
            "Make a slight forward move.",
          ],
        },
        zhenshchina: {
          title: "How to show “Woman”",
          text: "Sideways open palm: touch the right cheek, then the left cheek.",
          steps: [
            "Open the palm with fingers together and rotate the hand sideways (palm edge toward camera).",
            "Touch (or get very close to) the right cheek.",
            "Move to the left cheek and touch again. Keep the motion calm and readable.",
            "Tip: if your palm faces the camera, detection is worse — keep it sideways.",
          ],
        },
        solntse: {
          title: "How to show “Sun”",
          text: "Open palm above the head like sun rays.",
          steps: [
            "Lift one open palm above head level.",
            "Spread fingers slightly like rays.",
            "Make a small circular wrist motion.",
          ],
        },
        ya: {
          title: "How to show “I”",
          text: "Point to yourself in the chest area.",
          steps: [
            "Raise one hand with an index finger.",
            "Point to yourself in the center of chest.",
            "Hold the sign shortly and clearly.",
          ],
        },
        ty: {
          title: "How to show “You”",
          text: "Point at the other person with your index finger.",
          steps: [
            "Raise one hand with an index finger.",
            "Point the finger forward toward a person.",
            "Make a short, stable hold.",
          ],
        },
        horosho: {
          title: "How to show “Good”",
          text: "Classic thumbs-up gesture.",
          steps: [
            "Close one hand into a fist.",
            "Lift the thumb up.",
            "Hold the sign steadily for the camera.",
          ],
        },
        stop: {
          title: "How to show “Stop”",
          text: "Open palm facing forward, like “stop”.",
          steps: [
            "Raise one hand.",
            "Open your palm and face it forward.",
            "Keep the palm steady without waving.",
          ],
        },
        est: {
          title: "How to show “Eat”",
          text: "Bring fingers toward mouth several times, like eating.",
          steps: [
            "Shape fingers like a pinch.",
            "Move hand to mouth 2-3 times.",
            "Keep motions short and clear.",
          ],
        },
        pit: {
          title: "How to show “Drink”",
          text: "Pretend to hold a cup and bring it to mouth.",
          steps: [
            "Form hand like you hold a glass.",
            "Bring the hand to mouth.",
            "Repeat 1-2 times.",
          ],
        },
        spat: {
          title: "How to show “Sleep”",
          text: "Palm to cheek with a head tilt.",
          steps: [
            "Place palm near cheek.",
            "Tilt head toward the palm.",
            "Hold briefly.",
          ],
        },
        idti: {
          title: "How to show “Walk”",
          text: "Two fingers make walking steps in the air.",
          steps: [
            "Show index and middle fingers.",
            "Make 2-3 walking steps.",
            "Keep speed calm and even.",
          ],
        },
        bezhat: {
          title: "How to show “Run”",
          text: "Same two-finger gesture, but faster.",
          steps: [
            "Show index and middle fingers.",
            "Make fast and active stepping motion.",
            "Keep the direction stable.",
          ],
        },
        dumat: {
          title: "How to show “Think”",
          text: "Index finger touches temple.",
          steps: [
            "Raise index finger.",
            "Touch temple.",
            "Hold or make a tiny motion.",
          ],
        },
        lyubov: {
          title: "How to show “Love”",
          text: "Use a soft, symmetric self-hug: elbows point down and slightly outward, then hold for 1-2 seconds.",
          steps: [
            "Raise both hands to chest.",
            "Cross arms over chest.",
            "Hold for 1-2 seconds without jerky motion.",
          ],
        },
        bolshoy: {
          title: "How to show “Big”",
          text: "Spread both hands wide to show large size or volume.",
          steps: [
            "Raise both hands in front of you around chest level.",
            "Move them wide apart as if showing a big object.",
            "Hold the wide shape briefly without crossing arms.",
          ],
        },
        malenkiy: {
          title: "How to show “Small”",
          text: "Keep hands close together to show a small volume, or make a pinch gesture.",
          steps: [
            "Hold both hands in front of you with only a small gap between them.",
            "Slightly curve the fingers as if showing a narrow volume.",
            "Or make a pinch gesture with fingers and hold it briefly.",
          ],
        },
        krasiviy: {
          title: "How to show “Beautiful”",
          text: "Move an open palm smoothly along the face as if outlining the face oval.",
          steps: [
            "Bring an open palm near the face, with fingers together or slightly relaxed.",
            "Move the palm smoothly downward or in a gentle arc along the face outline.",
            "Finish with a soft pinch-like close or an elegant turn away.",
          ],
        },
        spasibo: {
          title: "How to show “Thank you”",
          text: "Make a fist and touch the forehead only. Show it frame-by-frame: touch → hold → release.",
          steps: [
            "Close one hand into a fist with the thumb on top.",
            "Raise the fist to the forehead and touch (or get very close).",
            "Hold the touch for 0.3–0.8 seconds — this is the key frame.",
            "Release calmly (no wide side-to-side waving, no repeats).",
          ],
        },
      },
      phraseChecklistItems: {
        da: ["Fist in front of chest", "Do 2-3 short up-down moves", "Keep motion small and clear"],
        net: ["Extend index and middle", "Thumb under them", "Quickly close fingers 1-2 times"],
        privet: ["Hand above shoulder", "Open palm", "Light side-to-side motion"],
        poka: ["Open palm", "Raise hand", "Turn wrist left-right"],
        dom: ["Show both hands", "Build a roof shape", "Hold the shape for 1 second"],
        druzhba: ["Make fists with both hands", "Interlock fists at chest level", "Or cross index fingers"],
        muzhchina: ["Open hand or index finger", "Touch forehead", "Move slightly forward"],
        zhenshchina: ["Touch cheek or chin", "Move downward", "Use one hand"],
        solntse: ["Show one hand", "Lift palm above head", "Make a small circular motion"],
        ya: ["Show one hand", "Point to yourself at chest center", "Hold the sign shortly"],
        ty: ["Show one hand", "Point index finger forward", "Hold direction shortly"],
        horosho: ["Show one hand", "Thumb up", "Keep the sign steady"],
        stop: ["Show one hand", "Open palm facing forward", "Keep palm stable, no wave"],
        est: ["Show one hand", "Bring fingers to mouth 2-3 times", "Use short amplitude"],
        pit: ["Show one hand", "Form a cup grip", "Bring to mouth 1-2 times"],
        spat: ["Show one hand", "Palm to cheek", "Tilt head to the palm"],
        idti: ["Show two fingers", "Make air walking steps", "Keep calm tempo"],
        bezhat: ["Show two fingers", "Make fast steps", "Use higher speed than walk"],
        dumat: ["Show index finger", "Touch temple", "Hold briefly"],
        lyubov: ["Show both hands at chest level", "Make a symmetric self-hug", "Hold 1-2 seconds without jerks"],
        bolshoy: ["Show both hands", "Spread them wide apart", "Hold the large volume shape"],
        malenkiy: ["Show hands close together", "Keep a small gap", "Or make a pinch gesture"],
        krasiviy: ["Show one palm near face", "Move smoothly along the face", "Finish softly"],
        spasibo: ["Fist", "Forehead touch", "Brief hold"],
      },
    },
    voice: {
      badge: "Voice Translation",
      title: "Speak and the app will recognize",
      text: "Record your voice and the app will transform speech into text.",
      start: "Start recording",
      stop: "Stop recording",
      recording: "Recording",
      transcript: "Recorded text",
      transcriptEmpty: "The text will appear here after recording.",
      transcriptHint: "After recognition, you can read the result and use it further.",
      micLabel: "Microphone button",
    },
    settings: {
      badge: "Settings",
      title: "Theme and language",
      text: "You can switch the look and the interface language here.",
      uiModeTitle: "UI mode",
      uiModeText: "Switch between Admin and User modes.",
      autoSpeakTitle: "Auto speak",
      autoSpeakText: "After recognition, the app will speak recognized words out loud.",
      themeTitle: "Theme",
      themeText: "Choose between light and dark appearance.",
      languageTitle: "Language",
      languageText: "Choose the interface language.",
      previewTitle: "Current setup",
      previewText: "Changes apply immediately to the whole interface.",
    },
    options: {
      theme: { dark: "Dark", light: "Light" },
      language: { ru: "Russian", en: "English" },
      autoSpeak: { on: "On", off: "Off" },
      uiMode: { admin: "Admin", user: "User" },
    },
    speech: {
      ready: "Tap the button and start speaking.",
      unsupported: "Speech recognition is not available in this browser.",
      listening: "Listening. Start speaking into the microphone.",
      transforming: "Your speech is being turned into text.",
      finishedSaved: "Recording finished. The text was saved.",
      stopped: "Recording stopped.",
      stopping: "Stopping the recording...",
      languageChanged: "Language changed. You can start recording in the new language now.",
      startError: "Could not start recording. Please check microphone access.",
      errors: {
        "audio-capture": "No microphone was found.",
        network: "A network connection is required for speech recognition.",
        "no-speech": "No speech was detected. Please try again.",
        "not-allowed": "Please allow microphone access.",
        "service-not-allowed": "The browser blocked the speech recognition service.",
        unknown: "Speech could not be recognized. Please try again.",
      },
    },
  },
};

const tabs = [
  { id: "home", Icon: HomeIcon },
  { id: "camera", Icon: CameraIcon },
  { id: "voice", Icon: VoiceIcon },
  { id: "settings", Icon: SettingsIcon },
];

const LIVE_WINDOW = 3;
const LIVE_MIN_INTERVAL_MS = 120;
const LIVE_SPEECH_MIN_INTERVAL_MS = 500;
const SIGN_PREDICTION_STICKY_MS = 1100;
const PHRASE_FAST_PROMOTE_CONFIDENCE = 0.9;
const PHRASE_FAST_PROMOTE_MARGIN = 0.18;
const PHRASE_FAST_PROMOTE_WEIGHTED_RATIO = 0.72;
const PHRASE_SIGN_FALLBACK_CANDIDATES = [
  {
    profile: "phrase_pack_target9_smart_v7",
    sourceModel: "sign-phrase-pack-target9-smart-v7",
    allowedLabelKeys: [
      "sign::SIGN_USER_DA",
      "sign::SIGN_PHRASEPACK_DOM",
      "sign::SIGN_PHRASEPACK_DRUZHBA",
      "sign::SIGN_PHRASEPACK_ZHENSHCHINA",
      "sign::SIGN_PHRASEPACK_MUZHCHINA",
      "sign::SIGN_USER_NET",
      "sign::SIGN_USER_POKA",
      "sign::SIGN_USER_PRIVET",
      "sign::SIGN_PHRASEPACK_SOLNTSE",
    ],
    minConfidence: 0.56,
    minMargin: 0.08,
  },
];
const SIGN_PRIVET_ALLOWED_LABEL_KEYS = [
  "sign::SIGN_HI",
  "sign::SIGN_USER_PRIVET",
];
const ALLOWED_WORD_LABELS = new Set(["Привет", "Пока", "Я", "Мужчина", "Женщина", "Спасибо"]);
const ALLOWED_PHRASE_IDS = new Set(["privet", "poka", "ya", "muzhchina", "zhenshchina", "spasibo"]);
const ACTIVE_SIGN_LABELS = new Set(ALLOWED_WORD_LABELS);
const PHRASE_LABEL_BY_ID = {
  privet: "Привет",
  poka: "Пока",
  ya: "Я",
  muzhchina: "Мужчина",
  zhenshchina: "Женщина",
  spasibo: "Спасибо",
};

const FACE_AREA_SIGN_IDS = new Set(["muzhchina", "zhenshchina", "spasibo"]);
const PHRASE_ID_BY_LABEL = Object.fromEntries(
  Object.entries(PHRASE_LABEL_BY_ID).map(([id, label]) => [label.toLowerCase(), id]),
);

function slugifyPhraseId(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9а-яё]+/gi, "-")
    .replace(/^-+|-+$/g, "");
}

function buildGenericPhraseGuide(label, phrase = {}) {
  const normalizedLabel = String(label ?? "").trim();
  const description = String(phrase.description ?? "").trim();
  const referenceNotes = String(phrase.referenceNotes ?? "").trim();
  const details = referenceNotes || description;

  return {
    title: `Как показать «${normalizedLabel}»`,
    text:
      details ||
      "Показывайте жест спокойно, с руками, лицом и верхней частью корпуса в кадре.",
    steps: details
      ? [details]
      : [
          "Покажите жест спокойно и без резких рывков.",
          "Держите руки, лицо и верхнюю часть корпуса в кадре.",
          "Повторите жест ещё раз, если камера не подтвердила распознавание.",
        ],
  };
}

function buildPhraseLibraryItemsFromServer(phrases = [], localeCopy) {
  return phrases
    .filter((item) => item.recognitionLevel === "phrase")
    .map((phrase, index) => {
      const normalizedLabel = String(phrase.label ?? phrase.textValue ?? "").trim();
      const knownPhraseId = PHRASE_ID_BY_LABEL[normalizedLabel.toLowerCase()];

      if (!ALLOWED_WORD_LABELS.has(normalizedLabel)) {
        return null;
      }

      if (knownPhraseId && !ALLOWED_PHRASE_IDS.has(knownPhraseId)) {
        return null;
      }

      const phraseId =
        knownPhraseId ||
        slugifyPhraseId(phrase.unitCode || normalizedLabel) ||
        `phrase-${phrase.id || index + 1}`;
      const localeGuide = localeCopy.camera.phraseGuideItems?.[knownPhraseId];
      const localeChecklist = localeCopy.camera.phraseChecklistItems?.[knownPhraseId];

      return {
        id: phraseId,
        label: normalizedLabel,
        aliases: [normalizedLabel],
        guide: localeGuide || buildGenericPhraseGuide(normalizedLabel, phrase),
        checklist: localeChecklist || [
          "Руки и верхняя часть корпуса в кадре",
          "Жест выполняется спокойно и читаемо",
          "При необходимости повторите жест ещё раз",
        ],
      };
    })
    .filter(Boolean);
}

function isModelNotTrainedError(error) {
  const message = String(error?.message ?? "").toLowerCase();
  return message.includes("not trained");
}

function getSpeechRecognitionApi() {
  if (typeof window === "undefined") {
    return null;
  }

  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function getVoiceStatusText(localeCopy, statusKey) {
  if (statusKey.startsWith("errors.")) {
    const errorCode = statusKey.slice("errors.".length);
    return localeCopy.speech.errors[errorCode] ?? localeCopy.speech.errors.unknown;
  }

  return localeCopy.speech[statusKey] ?? localeCopy.speech.ready;
}

function summarizeLiveWindow(entries) {
  if (!entries.length) {
    return { label: "", ratio: 0, weightedRatio: 0, averageConfidence: 0, averageMargin: 0 };
  }

  const stats = entries.reduce((acc, entry) => {
    if (!entry.label) {
      return acc;
    }

    const current = acc[entry.label] ?? { count: 0, weight: 0 };
    current.count += 1;
    current.weight += 0.2 + entry.confidence * 0.65 + entry.margin * 1.8;
    acc[entry.label] = current;
    return acc;
  }, {});

  const ranked = Object.entries(stats).sort(
    (left, right) => right[1].weight - left[1].weight || right[1].count - left[1].count,
  );
  const [label, dominant] = ranked[0] ?? ["", { count: 0, weight: 0 }];
  const totalWeight = ranked.reduce((sum, [, value]) => sum + value.weight, 0);
  const averageConfidence = entries.reduce((sum, entry) => sum + entry.confidence, 0) / entries.length;
  const averageMargin = entries.reduce((sum, entry) => sum + entry.margin, 0) / entries.length;

  return {
    label,
    ratio: dominant.count / entries.length,
    weightedRatio: totalWeight > 0 ? dominant.weight / totalWeight : 0,
    averageConfidence,
    averageMargin,
  };
}

const SIGN_LIVE_THRESHOLDS = {
  "Привет": {
    minEntries: 2,
    minRatio: 0.46,
    minWeightedRatio: 0.46,
    minConfidence: 0.5,
    minMargin: 0.02,
  },
  "Пока": {
    minEntries: 2,
    minRatio: 0.5,
    minWeightedRatio: 0.5,
    minConfidence: 0.52,
    minMargin: 0.02,
  },
  "Я": {
    minEntries: 2,
    minRatio: 0.48,
    minWeightedRatio: 0.5,
    minConfidence: 0.5,
    minMargin: 0.02,
  },
  "Мужчина": {
    minEntries: 2,
    minRatio: 0.48,
    minWeightedRatio: 0.5,
    minConfidence: 0.5,
    minMargin: 0.02,
  },
  "Женщина": {
    minEntries: 2,
    minRatio: 0.48,
    minWeightedRatio: 0.5,
    minConfidence: 0.5,
    minMargin: 0.02,
  },
  "Спасибо": {
    minEntries: 3,
    minRatio: 0.56,
    minWeightedRatio: 0.58,
    minConfidence: 0.58,
    minMargin: 0.04,
  },
};

function getLivePromoteThresholds({
  recognitionLevel,
  label = "",
} = {}) {
  if (recognitionLevel === "alphabet") {
    return {
      minEntries: 3,
      minRatio: 0.6,
      minWeightedRatio: 0.58,
      minConfidence: 0.34,
      minMargin: 0.035,
    };
  }

  if (recognitionLevel === "sign") {
    return {
      minEntries: 3,
      minRatio: 0.5,
      minWeightedRatio: 0.48,
      minConfidence: 0.34,
      minMargin: 0.025,
      ...(SIGN_LIVE_THRESHOLDS[String(label ?? "").trim()] ?? {}),
    };
  }

  return {
    minEntries: 3,
    minRatio: 0.6,
    minWeightedRatio: 0.58,
    minConfidence: 0.42,
    minMargin: 0.07,
  };
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function average(values = []) {
  if (!Array.isArray(values) || !values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getPointDistance(firstPoint, secondPoint) {
  if (!firstPoint || !secondPoint) {
    return 0;
  }

  const deltaX = Number(firstPoint.x ?? 0) - Number(secondPoint.x ?? 0);
  const deltaY = Number(firstPoint.y ?? 0) - Number(secondPoint.y ?? 0);
  return Math.hypot(deltaX, deltaY);
}

function getFaceFeaturePoint(faceLandmarks = [], indices = []) {
  if (!Array.isArray(faceLandmarks) || !faceLandmarks.length) {
    return null;
  }

  const points = indices
    .map((index) => faceLandmarks?.[index])
    .filter(Boolean);

  if (!points.length) {
    return null;
  }

  return {
    x: average(points.map((point) => Number(point?.x ?? 0))),
    y: average(points.map((point) => Number(point?.y ?? 0))),
  };
}

function getHandCenter(landmarks = []) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return null;
  }

  const anchorIndices = [0, 5, 9, 13, 17];
  const points = anchorIndices
    .map((index) => landmarks?.[index])
    .filter(Boolean);

  if (!points.length) {
    return null;
  }

  return {
    x: average(points.map((point) => Number(point?.x ?? 0))),
    y: average(points.map((point) => Number(point?.y ?? 0))),
  };
}

function getShoulderFrameStats(poseLandmarks = []) {
  if (!Array.isArray(poseLandmarks) || !poseLandmarks.length) {
    return null;
  }

  const leftShoulder = poseLandmarks?.[11] ?? null;
  const rightShoulder = poseLandmarks?.[12] ?? null;

  if (leftShoulder && rightShoulder) {
    return {
      centerY:
        (Number(leftShoulder.y ?? 0) + Number(rightShoulder.y ?? 0)) / 2,
      scale: Math.max(getPointDistance(leftShoulder, rightShoulder), 0.08),
      hasPair: true,
    };
  }

  const shoulder = leftShoulder ?? rightShoulder;

  if (!shoulder) {
    return null;
  }

  return {
    centerY: Number(shoulder.y ?? 0),
    scale: 0.12,
    hasPair: false,
  };
}

function getHeadFrameStats(faceLandmarks = [], poseLandmarks = []) {
  const leftEye = getFaceFeaturePoint(faceLandmarks, [33, 133, 159, 145]);
  const rightEye = getFaceFeaturePoint(faceLandmarks, [362, 263, 386, 374]);
  const nose = getFaceFeaturePoint(faceLandmarks, [1, 4, 5]);
  const forehead =
    faceLandmarks?.[10] ??
    faceLandmarks?.[151] ??
    faceLandmarks?.[9] ??
    null;

  if (leftEye && rightEye) {
    const eyeCenter = {
      x: (Number(leftEye.x ?? 0) + Number(rightEye.x ?? 0)) / 2,
      y: (Number(leftEye.y ?? 0) + Number(rightEye.y ?? 0)) / 2,
    };
    const topY = forehead
      ? Number(forehead.y ?? eyeCenter.y)
      : eyeCenter.y - 0.09;
    const scale = Math.max(getPointDistance(leftEye, rightEye), 0.08);

    return {
      centerX: Number(nose?.x ?? eyeCenter.x),
      eyeY: eyeCenter.y,
      topY,
      scale,
      hasFace: true,
    };
  }

  const poseNose = poseLandmarks?.[0] ?? null;
  const leftEar = poseLandmarks?.[7] ?? null;
  const rightEar = poseLandmarks?.[8] ?? null;
  const shoulderStats = getShoulderFrameStats(poseLandmarks);

  if (!poseNose && !shoulderStats) {
    return null;
  }

  const centerX = Number(
    poseNose?.x ??
      (
        (Number(leftEar?.x ?? 0) + Number(rightEar?.x ?? 0)) / 2 ||
        0.5
      ),
  );
  const eyeY = Number(
    poseNose?.y ??
      (
        (Number(leftEar?.y ?? 0) + Number(rightEar?.y ?? 0)) / 2 ||
        0.34
      ),
  );
  const topY = eyeY - 0.09;
  const scale = Math.max(
    shoulderStats?.scale ?? getPointDistance(leftEar, rightEar) ?? 0.1,
    0.08,
  );

  return {
    centerX,
    eyeY,
    topY,
    scale,
    hasFace: false,
  };
}

function getChestTargetPoint(poseLandmarks = [], headStats = null) {
  const leftShoulder = poseLandmarks?.[11] ?? null;
  const rightShoulder = poseLandmarks?.[12] ?? null;
  const shoulders = [leftShoulder, rightShoulder].filter(Boolean);
  const shoulderStats = getShoulderFrameStats(poseLandmarks);

  if (shoulders.length) {
    const centerX = average(shoulders.map((point) => Number(point?.x ?? 0)));
    const centerY = average(shoulders.map((point) => Number(point?.y ?? 0)));
    const scale = Math.max(Number(shoulderStats?.scale ?? 0.12), 0.08);

    return {
      x: centerX,
      y: centerY + scale * 0.16,
      scale,
    };
  }

  if (headStats) {
    const scale = Math.max(Number(headStats.scale ?? 0.1), 0.08);

    return {
      x: Number(headStats.centerX ?? 0.5),
      y: Number(headStats.eyeY ?? 0.34) + scale * 1.02,
      scale,
    };
  }

  return null;
}

function countDirectionChanges(values = [], minDelta = 0.0018) {
  if (!Array.isArray(values) || values.length < 3) {
    return 0;
  }

  let previousDirection = 0;
  let changes = 0;

  for (let index = 1; index < values.length; index += 1) {
    const delta = Number(values[index] ?? 0) - Number(values[index - 1] ?? 0);

    if (Math.abs(delta) < minDelta) {
      continue;
    }

    const direction = delta > 0 ? 1 : -1;

    if (previousDirection && direction !== previousDirection) {
      changes += 1;
    }

    previousDirection = direction;
  }

  return changes;
}

function getDominantHandFrameSet(frames = []) {
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );
  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = frames.length ? oneHandFrames / frames.length : 0;

  return {
    dominantHand,
    handedness,
    handFrames,
    oneHandRatio,
  };
}

function getMouthTarget(faceLandmarks = [], poseLandmarks = []) {
  const lips = getFaceFeaturePoint(faceLandmarks, [13, 14, 78, 308]);
  const nose = getFaceFeaturePoint(faceLandmarks, [1, 4, 5]);

  if (lips) {
    return {
      x: Number(lips.x ?? 0.5),
      y: Number(lips.y ?? 0.5),
      scale: Math.max(getPointDistance(nose, lips), 0.08),
    };
  }

  const poseNose = poseLandmarks?.[0] ?? null;

  if (!poseNose) {
    return null;
  }

  return {
    x: Number(poseNose.x ?? 0.5),
    y: Number(poseNose.y ?? 0.35) + 0.08,
    scale: 0.12,
  };
}

function getPredictionMetrics(prediction) {
  const confidence = Math.max(0, Math.min(1, Number(prediction?.confidence ?? 0)));
  const scores = Array.isArray(prediction?.scores) ? prediction.scores : [];
  const margin = scores.length >= 2
    ? Math.max(0, Number(scores[0]?.confidence ?? 0) - Number(scores[1]?.confidence ?? 0))
    : 0;
  const label = String(prediction?.label ?? "").trim();

  return {
    confidence,
    scores,
    margin,
    label,
  };
}

function getSignDetectorAffinity(prediction) {
  const label = String(prediction?.label ?? "").trim();
  const detector = prediction?.detector ?? {};

  switch (label) {
    case "Привет":
      return clamp01(
        Number(detector.waveScore ?? 0) * 0.34 +
        clamp01((Number(detector.horizontalRange ?? 0) - 0.04) / 0.14) * 0.18 +
        clamp01((0.2 - Number(detector.horizontalRange ?? 0)) / 0.2) * 0.1 +
        clamp01((Number(detector.shoulderLevelRatio ?? 0) - 0.3) / 0.5) * 0.18 +
        clamp01((Number(detector.openPalmRatio ?? 0) - 0.2) / 0.6) * 0.12 +
        clamp01((2 - Math.max(0, Number(detector.directionChanges ?? 0) - 1)) / 2) * 0.08
      );
    case "Пока":
      return clamp01(
        clamp01((Number(detector.openPalmRatio ?? 0) - 0.14) / 0.7) * 0.18 +
        clamp01((Number(detector.closePalmRatio ?? 0) - 0.08) / 0.56) * 0.12 +
        clamp01((Number(detector.xDirectionChanges ?? detector.palmPulseTransitions ?? 0) - 1) / 3) * 0.22 +
        clamp01((Number(detector.xRange ?? 0) - 0.04) / 0.16) * 0.16 +
        (detector.waveLike ? 0.18 : 0) +
        (detector.pulseLike ? 0.14 : 0)
      );
    case "Я":
      return clamp01(
        clamp01((Number(detector.nearChestRatio ?? 0) - 0.14) / 0.6) * 0.26 +
        clamp01((Number(detector.chestTouchRatio ?? 0) - 0.08) / 0.42) * 0.24 +
        clamp01((Number(detector.inwardPointRatio ?? 0) - 0.36) / 0.5) * 0.22 +
        clamp01((0.72 - Number(detector.centerOffsetAverage ?? 1)) / 0.72) * 0.16 +
        (detector.stableSelfPoint ? 0.12 : 0)
      );
    case "Ты":
      return clamp01(
        clamp01((Number(detector.outwardRatio ?? 0) - 0.22) / 0.56) * 0.3 +
        clamp01((0.5 - Number(detector.selfLikeRatio ?? 1)) / 0.5) * 0.16 +
        clamp01((Number(detector.outwardMarginAverage ?? 0) - 0.08) / 0.24) * 0.18 +
        clamp01((Number(detector.distanceToChestAverage ?? 0) - 1) / 0.7) * 0.16 +
        (detector.stableDirection ? 0.12 : 0)
      );
    case "Мужчина":
      return clamp01(
        clamp01((Number(detector.nearRatio ?? 0) - 0.16) / 0.56) * 0.28 +
        clamp01((Number(detector.gripLikeRatio ?? 0) - 0.18) / 0.6) * 0.16 +
        clamp01((Number(detector.downwardDelta ?? 0) - 0.01) / 0.18) * 0.16 +
        clamp01((Number(detector.oneHandRatio ?? 0) - 0.28) / 0.56) * 0.12 +
        clamp01((Number(detector.proximityScore ?? 0) - 0.18) / 0.62) * 0.14 +
        clamp01((Number(detector.lipProximityScore ?? 0) - 0.18) / 0.62) * 0.14
      );
    case "Женщина":
      return clamp01(
        clamp01((Number(detector.nearCheekRatio ?? 0) - 0.12) / 0.64) * 0.3 +
        clamp01((Number(detector.lowerFaceRatio ?? 0) - 0.16) / 0.68) * 0.24 +
        (detector.downwardSweep ? 0.12 : 0) +
        (detector.sideSweep ? 0.12 : 0) +
        clamp01((Number(detector.oneHandRatio ?? 0) - 0.22) / 0.58) * 0.1
      );
    case "Большой":
      return clamp01(
        clamp01((Number(detector.spreadAverage ?? 0) - 1.22) / 1.48) * 0.34 +
        clamp01((Number(detector.wideRatio ?? 0) - 0.08) / 0.72) * 0.22 +
        clamp01((Number(detector.twoHandsRatio ?? 0) - 0.14) / 0.68) * 0.18 +
        clamp01((Number(detector.openHandsRatio ?? 0) - 0.12) / 0.7) * 0.14 +
        (detector.stableHold ? 0.12 : 0)
      );
    case "Маленький":
      return clamp01(
        clamp01((1.16 - Number(detector.gapAverage ?? 99)) / 1.16) * 0.34 +
        clamp01((Number(detector.closeHandsRatio ?? 0) - 0.08) / 0.76) * 0.2 +
        clamp01((Number(detector.parallelRatio ?? 0) - 0.06) / 0.66) * 0.12 +
        clamp01((Number(detector.pinchRatio ?? 0) - 0.08) / 0.68) * 0.2 +
        (detector.stableHold ? 0.14 : 0)
      );
    case "Красивый":
      return clamp01(
        clamp01((Number(detector.nearFaceRatio ?? 0) - 0.1) / 0.68) * 0.28 +
        clamp01((Number(detector.openPalmRatio ?? 0) - 0.08) / 0.68) * 0.16 +
        clamp01((1.08 - Number(detector.yRangeNorm ?? 0)) / 1.08) * 0.12 +
        (detector.downwardSweep ? 0.14 : 0) +
        (detector.softArc ? 0.16 : 0) +
        clamp01((Number(detector.gracefulEndRatio ?? 0) - 0.04) / 0.76) * 0.14
      );
    case "Спасибо":
      return clamp01(
        clamp01((Number(detector.fistRatio ?? 0) - 0.18) / 0.72) * 0.26 +
        clamp01((Number(detector.foreheadRatio ?? 0) - 0.14) / 0.76) * 0.36 +
        (detector.touchDetected ? 0.14 : 0) +
        clamp01((3.2 - Number(detector.xRangeNorm ?? 9)) / 3.2) * 0.12 +
        clamp01((3.2 - Number(detector.yRangeNorm ?? 9)) / 3.2) * 0.12
      );
    default:
      return clamp01(Number(detector.detectorScore ?? 0));
  }
}

function canInstantPromotePrediction({
  recognitionLevel,
  prediction,
  liveEntries,
  summary,
  isWaveGreetingSequence = false,
  focusLabel = "",
}) {
  const supportsPrivetFastPath =
    recognitionLevel === "phrase" || recognitionLevel === "sign";

  if (!supportsPrivetFastPath || !prediction) {
    return false;
  }

  const metrics = getPredictionMetrics(prediction);
  const detectorType = String(prediction?.detector?.type ?? "").trim();
  const normalizedFocusLabel = String(focusLabel ?? "").trim();
  const affinityScore = getSignDetectorAffinity(prediction);
  const hasDetectorFastPath =
    detectorType !== "" &&
    metrics.confidence >= 0.84 &&
    metrics.margin >= 0.12;

  if (isWaveGreetingSequence && metrics.label === "Привет" && metrics.confidence >= 0.76) {
    return true;
  }

  if (
    recognitionLevel === "sign" &&
    metrics.confidence >= 0.93 &&
    metrics.margin >= 0.2 &&
    affinityScore >= 0.7
  ) {
    return true;
  }

  if (liveEntries.length < 2 || !summary.label || summary.label !== metrics.label) {
    return false;
  }

  return (
    (metrics.confidence >= PHRASE_FAST_PROMOTE_CONFIDENCE &&
      metrics.margin >= PHRASE_FAST_PROMOTE_MARGIN &&
      summary.weightedRatio >= PHRASE_FAST_PROMOTE_WEIGHTED_RATIO) ||
    (hasDetectorFastPath &&
      summary.weightedRatio >= 0.66 &&
      summary.averageConfidence >= 0.74 &&
      summary.averageMargin >= 0.1) ||
    (
      recognitionLevel === "sign" &&
      normalizedFocusLabel &&
      metrics.label === normalizedFocusLabel &&
      affinityScore >= 0.56 &&
      summary.weightedRatio >= 0.5 &&
      summary.averageConfidence >= 0.58
    )
  );
}

function chooseFocusedPhrasePrediction({
  focusLabel = "",
  resolvedPrediction = null,
  candidates = [],
} = {}) {
  const normalizedFocusLabel = String(focusLabel ?? "").trim();

  if (!normalizedFocusLabel) {
    return resolvedPrediction;
  }

  const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
  const focusedCandidates = candidates.filter((prediction) => {
    const metrics = getPredictionMetrics(prediction);
    return metrics.label === normalizedFocusLabel;
  });

  if (!focusedCandidates.length) {
    return resolvedPrediction;
  }

  const bestFocused = focusedCandidates
    .map((prediction) => {
      const metrics = getPredictionMetrics(prediction);
      const detectorScore = Number(prediction?.detector?.detectorScore ?? 0);
      const affinityScore = getSignDetectorAffinity(prediction);
      const score = metrics.confidence * 0.66 + detectorScore * 0.14 + affinityScore * 0.2;

      return {
        prediction,
        confidence: metrics.confidence,
        affinityScore,
        score,
      };
    })
    .sort((left, right) => right.score - left.score)[0];

  if (!bestFocused?.prediction) {
    return resolvedPrediction;
  }

  const isBolshoyFocus =
    normalizedFocusLabel === "Большой" || normalizedFocusLabel === "Big";
  const isMalenkiyFocus =
    normalizedFocusLabel === "Маленький" || normalizedFocusLabel === "Small";
  const isYaFocus = normalizedFocusLabel === "Я" || normalizedFocusLabel === "I";
  const isTyFocus = normalizedFocusLabel === "Ты" || normalizedFocusLabel === "You";
  const isMuzhchinaFocus =
    normalizedFocusLabel === "Мужчина" || normalizedFocusLabel === "Man";
  const isZhenshchinaFocus =
    normalizedFocusLabel === "Женщина" || normalizedFocusLabel === "Woman";
  const isKrasiviyFocus =
    normalizedFocusLabel === "Красивый" || normalizedFocusLabel === "Beautiful";
  const isSpasiboFocus =
    normalizedFocusLabel === "Спасибо" || normalizedFocusLabel === "Thank you";
  const focusedSpreadAverage = Number(
    bestFocused?.prediction?.detector?.spreadAverage ?? 0,
  );
  const focusedTwoHandsRatio = Number(
    bestFocused?.prediction?.detector?.twoHandsRatio ?? 0,
  );
  const focusedGapAverage = Number(
    bestFocused?.prediction?.detector?.gapAverage ?? 99,
  );
  const focusedNearChestRatio = Number(
    bestFocused?.prediction?.detector?.nearChestRatio ?? 0,
  );
  const focusedChestTouchRatio = Number(
    bestFocused?.prediction?.detector?.chestTouchRatio ?? 0,
  );
  const focusedOutwardRatio = Number(
    bestFocused?.prediction?.detector?.outwardRatio ?? 0,
  );
  const focusedNearRatio = Number(
    bestFocused?.prediction?.detector?.nearRatio ?? 0,
  );
  const focusedNearCheekRatio = Number(
    bestFocused?.prediction?.detector?.nearCheekRatio ?? 0,
  );
  const focusedNearFaceRatio = Number(
    bestFocused?.prediction?.detector?.nearFaceRatio ?? 0,
  );
  const focusedForeheadRatio = Number(
    bestFocused?.prediction?.detector?.foreheadRatio ?? 0,
  );

  if (
    isBolshoyFocus &&
    focusedSpreadAverage >= 1.12 &&
    focusedTwoHandsRatio >= 0.14 &&
    resolvedMetrics.confidence < 0.96
  ) {
    return bestFocused.prediction;
  }

  if (
    isMalenkiyFocus &&
    focusedGapAverage <= 1.14 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isYaFocus &&
    focusedNearChestRatio >= 0.18 &&
    focusedChestTouchRatio >= 0.1 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isTyFocus &&
    focusedOutwardRatio >= 0.26 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isMuzhchinaFocus &&
    focusedNearRatio >= 0.18 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isZhenshchinaFocus &&
    focusedNearCheekRatio >= 0.14 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isKrasiviyFocus &&
    focusedNearFaceRatio >= 0.1 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (
    isSpasiboFocus &&
    focusedForeheadRatio >= 0.16 &&
    resolvedMetrics.confidence < 0.94
  ) {
    return bestFocused.prediction;
  }

  if (!resolvedMetrics.label || resolvedMetrics.label === normalizedFocusLabel) {
    return bestFocused.prediction;
  }

  if (resolvedMetrics.confidence >= 0.95) {
    return resolvedPrediction;
  }

  if (
    bestFocused.confidence >= resolvedMetrics.confidence - 0.02 ||
    bestFocused.affinityScore >= 0.48 ||
    (bestFocused.confidence >= 0.74 && resolvedMetrics.confidence < 0.9)
  ) {
    return bestFocused.prediction;
  }

  return resolvedPrediction;
}

function chooseBestActiveSignPrediction({
  resolvedPrediction = null,
  candidates = [],
  focusLabel = "",
} = {}) {
  const normalizedFocusLabel = String(focusLabel ?? "").trim();
  const activeCandidates = candidates
    .filter(Boolean)
    .filter(
      (prediction) =>
        !shouldRejectActiveSignPrediction({
          prediction,
          focusLabel: normalizedFocusLabel,
        }),
    )
    .map((prediction) => {
      const metrics = getPredictionMetrics(prediction);
      const label = String(metrics.label ?? "").trim();
      const detectorScore = Number(prediction?.detector?.detectorScore ?? 0);
      const confidence = metrics.confidence;
      const margin = metrics.margin;
      const affinityScore = getSignDetectorAffinity(prediction);

      if (!label || !ACTIVE_SIGN_LABELS.has(label)) {
        return null;
      }

      const focusBonus = normalizedFocusLabel && label === normalizedFocusLabel ? 0.08 : 0;
      return {
        prediction,
        label,
        confidence,
        detectorScore,
        margin,
        affinityScore,
        score: confidence * 0.58 + detectorScore * 0.16 + margin * 0.04 + affinityScore * 0.22 + focusBonus,
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.score - left.score);

  if (!activeCandidates.length) {
    return resolvedPrediction;
  }

  const bestActive = activeCandidates[0];
  const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
  const resolvedLabel = String(resolvedMetrics.label ?? "").trim();
  const resolvedIsActive =
    ACTIVE_SIGN_LABELS.has(resolvedLabel) &&
    !shouldRejectActiveSignPrediction({
      prediction: resolvedPrediction,
      focusLabel: normalizedFocusLabel,
    });

  if (!resolvedIsActive) {
    if (bestActive.confidence >= 0.64 || bestActive.detectorScore >= 0.3 || bestActive.affinityScore >= 0.44) {
      return bestActive.prediction;
    }
    return resolvedPrediction;
  }

  if (!resolvedLabel || resolvedLabel === bestActive.label) {
    return resolvedPrediction || bestActive.prediction;
  }

  if (
    bestActive.confidence >= resolvedMetrics.confidence + 0.02 ||
    bestActive.affinityScore >= 0.52 ||
    bestActive.score >= resolvedMetrics.confidence * 0.78 + 0.18 ||
    (bestActive.detectorScore >= 0.36 && resolvedMetrics.confidence < 0.84)
  ) {
    return bestActive.prediction;
  }

  return resolvedPrediction;
}

function shouldRejectActiveSignPrediction({
  prediction = null,
  focusLabel = "",
} = {}) {
  const metrics = getPredictionMetrics(prediction);
  const label = String(metrics.label ?? "").trim();

  if (!label || !ACTIVE_SIGN_LABELS.has(label)) {
    return false;
  }

  const detector = prediction?.detector ?? {};
  const normalizedFocusLabel = String(focusLabel ?? "").trim();
  const isFocused = normalizedFocusLabel && normalizedFocusLabel === label;
  const affinityScore = getSignDetectorAffinity(prediction);
  const minAffinity = isFocused ? 0.14 : 0.2;

  if (affinityScore < minAffinity && metrics.confidence < (isFocused ? 0.7 : 0.76)) {
    return true;
  }

  switch (label) {
    case "Привет": {
      const openPalmRatio = Number(detector.openPalmRatio ?? 0);
      const shoulderLevelRatio = Number(detector.shoulderLevelRatio ?? 0);
      const horizontalRange = Number(detector.horizontalRange ?? 0);
      const directionChanges = Number(detector.directionChanges ?? 0);
      const waveScore = Number(detector.waveScore ?? 0);

      return (
        openPalmRatio < (isFocused ? 0.08 : 0.12) ||
        shoulderLevelRatio < (isFocused ? 0.16 : 0.22) ||
        horizontalRange > 0.26 ||
        directionChanges > 4 ||
        (
          waveScore < (isFocused ? 0.12 : 0.18) &&
          metrics.confidence < 0.82
        )
      );
    }
    case "Пока": {
      const openPalmRatio = Number(detector.openPalmRatio ?? 0);
      const closePalmRatio = Number(detector.closePalmRatio ?? 0);
      const palmPulseTransitions = Number(detector.palmPulseTransitions ?? 0);
      const pulseLike = Boolean(detector.pulseLike);

      return (
        openPalmRatio < (isFocused ? 0.1 : 0.14) ||
        closePalmRatio < (isFocused ? 0.08 : 0.12) ||
        !pulseLike ||
        palmPulseTransitions < 1
      );
    }
    case "Я": {
      const nearChestRatio = Number(detector.nearChestRatio ?? 0);
      const chestTouchRatio = Number(detector.chestTouchRatio ?? 0);
      const inwardPointRatio = Number(detector.inwardPointRatio ?? 0);
      const centerOffsetAverage = Number(detector.centerOffsetAverage ?? 1);

      return (
        nearChestRatio < (isFocused ? 0.1 : 0.14) ||
        chestTouchRatio < (isFocused ? 0.05 : 0.08) ||
        inwardPointRatio < (isFocused ? 0.3 : 0.38) ||
        centerOffsetAverage > (isFocused ? 0.84 : 0.74)
      );
    }
    case "Ты": {
      const outwardRatio = Number(detector.outwardRatio ?? 0);
      const selfLikeRatio = Number(detector.selfLikeRatio ?? 1);
      const outwardMarginAverage = Number(detector.outwardMarginAverage ?? 0);
      const distanceToChestAverage = Number(detector.distanceToChestAverage ?? 0);

      return (
        outwardRatio < (isFocused ? 0.18 : 0.24) ||
        selfLikeRatio > (isFocused ? 0.62 : 0.52) ||
        outwardMarginAverage < (isFocused ? 0.05 : 0.08) ||
        distanceToChestAverage < (isFocused ? 0.86 : 0.96)
      );
    }
    case "Мужчина": {
      const nearRatio = Number(detector.nearRatio ?? 0);
      const proximityScore = Number(detector.proximityScore ?? 0);
      const oneHandRatio = Number(detector.oneHandRatio ?? 0);

      return (
        nearRatio < (isFocused ? 0.12 : 0.18) ||
        proximityScore < (isFocused ? 0.1 : 0.16) ||
        oneHandRatio < (isFocused ? 0.18 : 0.24)
      );
    }
    case "Женщина": {
      const nearCheekRatio = Number(detector.nearCheekRatio ?? 0);
      const lowerFaceRatio = Number(detector.lowerFaceRatio ?? 0);
      const sideSweep = Boolean(detector.sideSweep);
      const downwardSweep = Boolean(detector.downwardSweep);

      return (
        nearCheekRatio < (isFocused ? 0.08 : 0.12) ||
        lowerFaceRatio < (isFocused ? 0.12 : 0.16) ||
        (
          !sideSweep &&
          !downwardSweep &&
          nearCheekRatio < 0.18 &&
          metrics.confidence < 0.84
        )
      );
    }
    case "Большой": {
      const spreadAverage = Number(detector.spreadAverage ?? 0);
      const wideRatio = Number(detector.wideRatio ?? 0);
      const twoHandsRatio = Number(detector.twoHandsRatio ?? 0);

      return (
        spreadAverage < (isFocused ? 0.98 : 1.1) ||
        wideRatio < (isFocused ? 0.04 : 0.08) ||
        twoHandsRatio < (isFocused ? 0.08 : 0.14)
      );
    }
    case "Маленький": {
      const gapAverage = Number(detector.gapAverage ?? 99);
      const closeHandsRatio = Number(detector.closeHandsRatio ?? 0);
      const pinchRatio = Number(detector.pinchRatio ?? 0);

      return (
        gapAverage > (isFocused ? 1.32 : 1.18) &&
        closeHandsRatio < (isFocused ? 0.06 : 0.1) &&
        pinchRatio < (isFocused ? 0.06 : 0.1)
      );
    }
    case "Красивый": {
      const nearFaceRatio = Number(detector.nearFaceRatio ?? 0);
      const yRangeNorm = Number(detector.yRangeNorm ?? 0);
      const softArc = Boolean(detector.softArc);
      const downwardSweep = Boolean(detector.downwardSweep);
      const gracefulEndRatio = Number(detector.gracefulEndRatio ?? 0);

      return (
        nearFaceRatio < (isFocused ? 0.08 : 0.12) ||
        yRangeNorm > 1.18 ||
        (
          !softArc &&
          !downwardSweep &&
          gracefulEndRatio < (isFocused ? 0.02 : 0.04) &&
          metrics.confidence < 0.84
        )
      );
    }
    case "Спасибо": {
      const fistRatio = Number(detector.fistRatio ?? 0);
      const foreheadRatio = Number(detector.foreheadRatio ?? 0);
      const touchDetected = Boolean(detector.touchDetected);
      const xRangeNorm = Number(detector.xRangeNorm ?? 9);
      const yRangeNorm = Number(detector.yRangeNorm ?? 9);

      return (
        !touchDetected ||
        fistRatio < (isFocused ? 0.16 : 0.18) ||
        foreheadRatio < (isFocused ? 0.14 : 0.16) ||
        xRangeNorm > (isFocused ? 3.4 : 3.2) ||
        yRangeNorm > (isFocused ? 3.4 : 3.2)
      );
    }
    default:
      return false;
  }
}

function normalizePhraseSignPrediction(prediction, fallbackConfig = {}) {
  if (!prediction) {
    return null;
  }

  const rawLabel = String(prediction?.label ?? "").trim();
  const normalizedLabel = fallbackConfig.labelAliases?.[rawLabel] ?? rawLabel;

  return {
    ...prediction,
    label: normalizedLabel,
    sourceModel: fallbackConfig.sourceModel ?? prediction.sourceModel ?? "sign-phrase-fallback",
  };
}

function selectBestPhraseFallbackPrediction(fallbackEntries) {
  const candidates = fallbackEntries
    .map(({ prediction, config }) => {
      const normalizedPrediction = normalizePhraseSignPrediction(prediction, config);
      const { confidence, margin, label } = getPredictionMetrics(normalizedPrediction);

      if (
        !label ||
        confidence < Number(config?.minConfidence ?? 0.42) ||
        margin < Number(config?.minMargin ?? 0.08)
      ) {
        return null;
      }

      return {
        prediction: normalizedPrediction,
        confidence,
        margin,
        combinedScore: confidence + margin * 0.45,
      };
    })
    .filter(Boolean)
    .sort(
      (left, right) =>
        right.combinedScore - left.combinedScore ||
        right.confidence - left.confidence ||
        right.margin - left.margin,
    );

  return candidates[0]?.prediction ?? null;
}

function choosePhraseFallbackPrediction(primaryPrediction, fallbackEntries) {
  const fallbackPrediction = selectBestPhraseFallbackPrediction(fallbackEntries);

  if (!fallbackPrediction) {
    return primaryPrediction;
  }

  const primary = getPredictionMetrics(primaryPrediction);
  const fallback = getPredictionMetrics(fallbackPrediction);
  const primaryLooksReliable =
    primary.label &&
    primary.confidence >= 0.58 &&
    primary.margin >= 0.08;

  if (!primaryLooksReliable || primary.label === fallback.label) {
    return fallbackPrediction;
  }

  return primaryPrediction;
}

function isWaveGreetingMetadata(metadata) {
  return Boolean(
    metadata &&
      metadata.waveLike &&
      Number(metadata.waveScore ?? 0) >= 0.58 &&
      Number(metadata.waveOneHandRatio ?? 0) >= 0.6 &&
      Number(metadata.waveHandPresenceRatio ?? 0) >= 0.5,
  );
}

function isSimpleWaveHelloMetadata(metadata) {
  if (!metadata) {
    return false;
  }

  const horizontalRange = Number(metadata.waveHorizontalRange ?? 0);
  const directionChanges = Number(metadata.waveDirectionChanges ?? 0);
  const oneHandRatio = Number(metadata.waveOneHandRatio ?? 0);
  const handPresenceRatio = Number(metadata.waveHandPresenceRatio ?? 0);
  const horizontalDominance = Number(metadata.waveHorizontalDominance ?? 0);
  const waveScore = Number(metadata.waveScore ?? 0);

  return (
    horizontalRange >= 0.06 &&
    directionChanges >= 1 &&
    oneHandRatio >= 0.5 &&
    handPresenceRatio >= 0.4 &&
    (horizontalDominance === 0 || horizontalDominance >= 0.95) &&
    waveScore >= 0.44
  );
}

function buildPrivetTrajectoryPrediction({
  frames = [],
  metadata = null,
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8 || !isWaveGreetingMetadata(metadata)) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const dominantHand =
    String(metadata?.waveDominantHand ?? "right").toLowerCase() === "left"
      ? "left"
      : "right";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => (dominantHand === "left" ? frame?.left_hand : frame?.right_hand))
    .filter((landmarks) => Array.isArray(landmarks) && landmarks.length >= 21);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.45))) {
    return null;
  }

  const gesturePredictions = handFrames.map((landmarks) => predictGesture(landmarks, handedness));
  const openPalmFrames = gesturePredictions.filter(
    (prediction) => prediction?.gesture === "Открытая ладонь",
  );
  const openPalmStrongFrames = openPalmFrames.filter(
    (prediction) => Number(prediction?.confidence ?? 0) >= 0.48,
  );
  const openPalmRatio = openPalmFrames.length / gesturePredictions.length;
  const openPalmStrongRatio = openPalmStrongFrames.length / gesturePredictions.length;

  const featureSamples = handFrames
    .map((landmarks) => extractHandFeatures(landmarks, handedness))
    .filter(Boolean);
  const openPalmFeatureScore = average(
    featureSamples.map((features) => clamp01(
      (features.longFingerRelaxedCount / 4) * 0.44 +
      features.fingerExtensionAverage * 0.34 +
      features.palmSpread * 0.18 +
      (features.openScores?.thumb ?? 0) * 0.1,
    )),
  );

  const waveScore = Number(metadata?.waveScore ?? 0);
  const horizontalRange = Number(metadata?.waveHorizontalRange ?? 0);
  const horizontalDominance = Number(metadata?.waveHorizontalDominance ?? 0);
  const oneHandRatio = Number(metadata?.waveOneHandRatio ?? 0);
  const handPresenceRatio = Number(metadata?.waveHandPresenceRatio ?? 0);
  const directionChanges = Number(metadata?.waveDirectionChanges ?? 0);
  const shoulderLevelSamples = frames
    .map((frame) => {
      const pose = Array.isArray(frame?.pose) ? frame.pose : [];
      const shoulderStats = getShoulderFrameStats(pose);
      const activeHand =
        dominantHand === "left" ? frame?.left_hand : frame?.right_hand;

      if (!shoulderStats || !Array.isArray(activeHand) || activeHand.length < 21) {
        return null;
      }

      const handCenter = getHandCenter(activeHand);

      if (!handCenter) {
        return null;
      }

      return handCenter.y <= Number(shoulderStats.centerY ?? 0.5) + Number(shoulderStats.scale ?? 0.1) * 0.22;
    })
    .filter((value) => typeof value === "boolean");
  const shoulderLevelRatio = shoulderLevelSamples.length
    ? shoulderLevelSamples.filter(Boolean).length / shoulderLevelSamples.length
    : 0;
  const strictMotionGate =
    waveScore >= 0.54 &&
    horizontalRange >= 0.05 &&
    horizontalDominance >= 0.92 &&
    directionChanges >= 1 &&
    directionChanges <= 2 &&
    oneHandRatio >= 0.54 &&
    shoulderLevelRatio >= 0.42 &&
    handPresenceRatio >= 0.46;
  const strictPalmGate =
    openPalmStrongRatio >= 0.28 ||
    openPalmRatio >= 0.42 ||
    openPalmFeatureScore >= 0.5;
  const shortGreetingMotion =
    horizontalRange <= 0.18 &&
    waveScore <= 0.82;

  if (!strictMotionGate || !strictPalmGate || !shortGreetingMotion) {
    return null;
  }

  const detectorScore = clamp01(
    clamp01((waveScore - 0.54) / 0.32) * 0.34 +
    clamp01((horizontalRange - 0.05) / 0.11) * 0.14 +
    clamp01((horizontalDominance - 0.92) / 0.98) * 0.12 +
    clamp01(directionChanges / 2) * 0.08 +
    clamp01((oneHandRatio - 0.52) / 0.3) * 0.08 +
    clamp01((shoulderLevelRatio - 0.4) / 0.46) * 0.12 +
    clamp01((handPresenceRatio - 0.44) / 0.34) * 0.06 +
    clamp01((openPalmStrongRatio - 0.24) / 0.46) * 0.08 +
    clamp01((openPalmRatio - 0.36) / 0.4) * 0.04 +
    clamp01((openPalmFeatureScore - 0.48) / 0.32) * 0.16,
  );

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Привет" ||
    current.confidence < 0.68 ||
    (current.label === "Пока" && current.confidence < 0.82 && detectorScore >= 0.88);

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Привет" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Привет" ? current.label : "Пока";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.72 - detectorScore * 0.16),
  );

  return {
    label: "Привет",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-privet-detector",
    scores: [
      { label: "Привет", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "privet_wave_trajectory",
      detectorScore,
      waveScore,
      openPalmRatio,
      openPalmStrongRatio,
      openPalmFeatureScore,
      directionChanges,
      horizontalRange,
      horizontalDominance,
      oneHandRatio,
      shoulderLevelRatio,
      handPresenceRatio,
      dominantHand,
    },
  };
}

function buildPokaPalmPulsePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const directionSign = dominantHand === "left" ? -1 : 1;
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter((frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.45))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let openPalmCount = 0;
  let closePalmCount = 0;
  let nearTempleCount = 0;
  const xSamples = [];
  const outwardSamples = [];
  const palmStates = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const prediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? center;
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    const openPalmLike =
      (
        prediction?.gesture === "Открытая ладонь" &&
        Number(prediction?.confidence ?? 0) >= 0.26
      ) ||
      Number(features?.fingerExtensionAverage ?? 0) >= 0.42 ||
      Number(features?.longFingerRelaxedCount ?? 0) >= 2;
    const closePalmLike =
      (
        prediction?.gesture === "Кулак" &&
        Number(prediction?.confidence ?? 0) >= 0.22
      ) ||
      Number(features?.fingerExtensionAverage ?? 1) <= 0.34 ||
      Number(features?.curledLongFingerCount ?? 0) >= 2;

    if (openPalmLike) {
      openPalmCount += 1;
    }
    if (closePalmLike) {
      closePalmCount += 1;
    }

    if (!center || !indexTip || !noSecondHand) {
      continue;
    }

    const face = Array.isArray(frame?.face) ? frame.face : [];
    const headStats = getHeadFrameStats(face, frame?.pose);
    const leftEye = getFaceFeaturePoint(face, [33, 133, 159, 145]);
    const rightEye = getFaceFeaturePoint(face, [362, 263, 386, 374]);
    const sideEye = dominantHand === "left" ? leftEye : rightEye;
    const scale = Math.max(Number(headStats?.scale ?? 0.08), 0.08);
    const templePoint = sideEye
      ? {
          x: Number(sideEye.x ?? 0) + directionSign * scale * 0.28,
          y: Number(sideEye.y ?? 0) - scale * 0.02,
        }
      : {
          x: Number(headStats?.centerX ?? 0.5) + directionSign * scale * 0.34,
          y: Number(headStats?.eyeY ?? 0.34),
        };
    const templeDistance = getPointDistance(indexTip, templePoint) / scale;
    const nearTemple =
      openPalmLike &&
      templeDistance <= 1.12 &&
      Math.abs(Number(indexTip.y ?? 0) - templePoint.y) / scale <= 0.98;

    if (nearTemple) {
      nearTempleCount += 1;
    }

    if (openPalmLike) {
      palmStates.push("open");
    } else if (closePalmLike) {
      palmStates.push("close");
    }

    if (openPalmLike) {
      xSamples.push(Number(center.x ?? 0));
      if (headStats) {
        const outwardValue =
          ((Number(center.x ?? 0.5) - Number(headStats.centerX ?? 0.5)) * directionSign) / scale;
        outwardSamples.push(outwardValue);
      }
    }
  }

  if (xSamples.length < 4) {
    return null;
  }

  const openPalmRatio = openPalmCount / handFrames.length;
  const closePalmRatio = closePalmCount / handFrames.length;
  const nearTempleRatio = nearTempleCount / handFrames.length;
  const xDirectionChanges = countDirectionChanges(xSamples, 0.0018);
  const xRange = Math.max(...xSamples) - Math.min(...xSamples);
  const chunkSize = Math.max(2, Math.round(outwardSamples.length * 0.35));
  const startOutward = outwardSamples.length
    ? average(outwardSamples.slice(0, chunkSize))
    : 0;
  const endOutward = outwardSamples.length
    ? average(outwardSamples.slice(-chunkSize))
    : 0;
  const outwardDelta = endOutward - startOutward;
  let palmPulseTransitions = 0;
  let previousPalmState = palmStates[0] ?? "";

  for (let index = 1; index < palmStates.length; index += 1) {
    const nextPalmState = palmStates[index];

    if (nextPalmState && nextPalmState !== previousPalmState) {
      palmPulseTransitions += 1;
      previousPalmState = nextPalmState;
    }
  }

  const pulseLike =
    palmStates.includes("open") &&
    palmStates.includes("close") &&
    palmPulseTransitions >= 1;
  const pulseScore = clamp01(
    clamp01((oneHandRatio - 0.28) / 0.66) * 0.18 +
    clamp01((openPalmRatio - 0.14) / 0.82) * 0.24 +
    clamp01((closePalmRatio - 0.08) / 0.72) * 0.12 +
    clamp01((nearTempleRatio - 0.1) / 0.76) * 0.16 +
    clamp01(xDirectionChanges / 3) * 0.16 +
    clamp01((xRange - 0.04) / 0.24) * 0.14 +
    clamp01((outwardDelta - 0.06) / 0.46) * 0.08 +
    clamp01(palmPulseTransitions / 3) * 0.06,
  );
  const strictGate =
    oneHandRatio >= 0.24 &&
    openPalmRatio >= 0.14 &&
    pulseLike &&
    closePalmRatio >= 0.12;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Пока" ||
    current.confidence < 0.78 ||
    (
      ["Привет", "Женщина", "Солнце", "Дом", "Да", "Нет"].includes(current.label) &&
      current.confidence < 0.88
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Пока" ? current.confidence : 0,
    0.68 + pulseScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Пока" ? current.label : "Привет";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - pulseScore * 0.16),
  );

  return {
    label: "Пока",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-poka-palm-pulse-detector",
    scores: [
      { label: "Пока", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "poka_palm_pulse",
      detectorScore: pulseScore,
      openPalmRatio,
      closePalmRatio,
      nearTempleRatio,
      xDirectionChanges,
      xRange,
      outwardDelta,
      pulseLike,
      palmPulseTransitions,
      oneHandRatio,
      dominantHand,
    },
  };
}

function buildDaNodPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.45))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let fistCount = 0;
  let nodPoseCount = 0;
  let poseSupportedFrames = 0;
  let chestLevelFrames = 0;
  const nodSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const center = getHandCenter(landmarks);

    if (!center) {
      continue;
    }

    const gesturePrediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const fistDetected =
      (
        gesturePrediction?.gesture === "Кулак" &&
        Number(gesturePrediction?.confidence ?? 0) >= 0.38
      ) ||
      (
        Number(features?.curledLongFingerCount ?? 0) >= 3 &&
        Number(features?.fingerExtensionAverage ?? 1) <= 0.5
      );

    if (fistDetected) {
      fistCount += 1;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(shoulderStats?.scale ?? 0.12, 0.08);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (shoulderStats) {
      poseSupportedFrames += 1;
      const minChestY = shoulderStats.centerY - 0.14;
      const maxChestY = shoulderStats.centerY + 0.38;
      const isChestLevel = center.y >= minChestY && center.y <= maxChestY;

      if (isChestLevel) {
        chestLevelFrames += 1;
      }

      if (fistDetected && noSecondHand && isChestLevel) {
        nodPoseCount += 1;
        nodSamples.push({
          x: Number(center.x ?? 0),
          y: Number(center.y ?? 0),
          scale: frameScale,
        });
      }
    } else if (fistDetected && noSecondHand) {
      nodSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
        scale: frameScale,
      });
    }
  }

  if (nodSamples.length < 4) {
    return null;
  }

  const fistRatio = fistCount / handFrames.length;
  const nodPoseRatio = nodPoseCount / Math.max(handFrames.length, 1);
  const chestRatio =
    poseSupportedFrames > 0 ? chestLevelFrames / poseSupportedFrames : 1;
  const scaleAverage = Math.max(
    average(nodSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = nodSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = nodSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const xDirectionChanges = countDirectionChanges(xValues, 0.0014);
  const yDirectionChanges = countDirectionChanges(yValues, 0.0014);
  const verticalDominance = yRangeNorm / Math.max(xRangeNorm, 0.03);
  const nodLikeMotion =
    yRangeNorm >= 0.08 &&
    yRangeNorm <= 0.74 &&
    yDirectionChanges >= 1 &&
    verticalDominance >= 1.1 &&
    xRangeNorm <= 0.42;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.46) / 0.48) * 0.18 +
    clamp01((fistRatio - 0.34) / 0.64) * 0.24 +
    clamp01((nodPoseRatio - 0.22) / 0.68) * 0.16 +
    clamp01((chestRatio - 0.28) / 0.68) * 0.08 +
    clamp01((yRangeNorm - 0.07) / 0.5) * 0.16 +
    clamp01((0.36 - xRangeNorm) / 0.36) * 0.06 +
    clamp01((verticalDominance - 1.02) / 1.6) * 0.08 +
    clamp01(yDirectionChanges / 3) * 0.04,
  );
  const strictGate =
    oneHandRatio >= 0.5 &&
    fistRatio >= 0.4 &&
    nodPoseRatio >= 0.2 &&
    (poseSupportedFrames < 3 || chestRatio >= 0.34) &&
    nodLikeMotion;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Да" ||
    current.confidence < 0.82 ||
    (
      ["Нет", "Пока", "Привет", "Солнце", "Дом", "Женщина", "Мужчина", "Дружба"].includes(
        current.label,
      ) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Да" ? current.confidence : 0,
    0.68 + detectorScore * 0.27,
  ));
  const secondaryLabel =
    current.label && current.label !== "Да" ? current.label : "Нет";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Да",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-da-nod-detector",
    scores: [
      { label: "Да", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "da_fist_vertical_nod",
      detectorScore,
      oneHandRatio,
      fistRatio,
      nodPoseRatio,
      chestRatio,
      xRangeNorm,
      yRangeNorm,
      xDirectionChanges,
      yDirectionChanges,
      verticalDominance,
      nodLikeMotion,
      dominantHand,
    },
  };
}

function buildNetSnapPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.45))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let twoFingerCount = 0;
  let compactTwoFingerCount = 0;
  let tightTwoFingerCount = 0;
  let closeCount = 0;
  let chestLevelCount = 0;
  const stateFrames = [];
  const motionSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(shoulderStats?.scale ?? 0.12, 0.08);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const twoFingerState =
      (
        prediction?.gesture === "Два пальца" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(features?.openScores?.index ?? 0) >= 0.6 &&
        Number(features?.openScores?.middle ?? 0) >= 0.58 &&
        Number(features?.openScores?.ring ?? 1) <= 0.45 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.45
      );
    const compactTwoFinger =
      twoFingerState &&
      Number(features?.indexMiddleGap ?? 1) <= 0.32 &&
      Number(features?.indexMiddleGap ?? 0) >= 0.08 &&
      Number(features?.middleRingGap ?? 1) <= 0.26 &&
      Number(features?.ringPinkyGap ?? 1) <= 0.26;
    const tightTwoFinger =
      twoFingerState &&
      Number(features?.indexMiddleGap ?? 1) <= 0.22 &&
      Number(features?.indexMiddleGap ?? 0) >= 0.04 &&
      Number(features?.middleRingGap ?? 1) <= 0.22 &&
      Number(features?.ringPinkyGap ?? 1) <= 0.22;
    const closeState =
      (
        prediction?.gesture === "Кулак" &&
        Number(prediction?.confidence ?? 0) >= 0.32
      ) ||
      (
        Number(features?.curledLongFingerCount ?? 0) >= 2 &&
        Number(features?.fingerExtensionAverage ?? 1) <= 0.5
      ) ||
      (
        Number(features?.openScores?.index ?? 1) <= 0.42 &&
        Number(features?.openScores?.middle ?? 1) <= 0.42
      );

    if (twoFingerState) {
      twoFingerCount += 1;
    }
    if (compactTwoFinger) {
      compactTwoFingerCount += 1;
    }
    if (tightTwoFinger) {
      tightTwoFingerCount += 1;
    }

    if (closeState) {
      closeCount += 1;
    }

    if (noSecondHand) {
      if (tightTwoFinger) {
        stateFrames.push("two");
      } else if (closeState) {
        stateFrames.push("close");
      }
    }

    if (center && noSecondHand) {
      if (shoulderStats) {
        const minChestY = shoulderStats.centerY - 0.16;
        const maxChestY = shoulderStats.centerY + 0.34;
        const isChestLevel = center.y >= minChestY && center.y <= maxChestY;

        if (isChestLevel) {
          chestLevelCount += 1;
        }
      }
      motionSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
        scale: frameScale,
      });
    }
  }

  if (stateFrames.length < 5 || motionSamples.length < 4) {
    return null;
  }

  let transitions = 0;
  let previousState = stateFrames[0];

  for (let index = 1; index < stateFrames.length; index += 1) {
    const nextState = stateFrames[index];

    if (nextState !== previousState) {
      transitions += 1;
      previousState = nextState;
    }
  }

  const twoFingerRatio = twoFingerCount / handFrames.length;
  const compactTwoFingerRatio = compactTwoFingerCount / handFrames.length;
  const tightTwoFingerRatio = tightTwoFingerCount / handFrames.length;
  const closeRatio = closeCount / handFrames.length;
  const chestLevelRatio = chestLevelCount / Math.max(handFrames.length, 1);
  const hasSnapCycle =
    transitions >= 1 &&
    stateFrames.includes("two") &&
    stateFrames.includes("close");
  const scaleAverage = Math.max(
    average(motionSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = motionSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = motionSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.48) / 0.44) * 0.16 +
    clamp01((twoFingerRatio - 0.18) / 0.62) * 0.14 +
    clamp01((compactTwoFingerRatio - 0.14) / 0.66) * 0.14 +
    clamp01((tightTwoFingerRatio - 0.1) / 0.68) * 0.1 +
    clamp01((closeRatio - 0.16) / 0.62) * 0.24 +
    clamp01(transitions / 3) * 0.14 +
    (hasSnapCycle ? 0.1 : 0) +
    clamp01((chestLevelRatio - 0.22) / 0.68) * 0.1 +
    clamp01((0.34 - xRangeNorm) / 0.34) * 0.06 +
    clamp01((0.34 - yRangeNorm) / 0.34) * 0.06,
  );
  const strictGate =
    oneHandRatio >= 0.52 &&
    twoFingerRatio >= 0.2 &&
    compactTwoFingerRatio >= 0.14 &&
    tightTwoFingerRatio >= 0.1 &&
    closeRatio >= 0.18 &&
    chestLevelRatio >= 0.24 &&
    hasSnapCycle;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Нет" ||
    current.confidence < 0.84 ||
    (
      ["Да", "Пока", "Привет", "Солнце", "Дом", "Женщина", "Мужчина", "Дружба"].includes(
        current.label,
      ) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Нет" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Нет" ? current.label : "Да";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Нет",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-net-snap-detector",
    scores: [
      { label: "Нет", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "net_two_fingers_snap_close",
      detectorScore,
      oneHandRatio,
      twoFingerRatio,
      compactTwoFingerRatio,
      tightTwoFingerRatio,
      closeRatio,
      chestLevelRatio,
      transitions,
      hasSnapCycle,
      xRangeNorm,
      yRangeNorm,
      dominantHand,
    },
  };
}

function buildYaPointSelfPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let pointingCount = 0;
  let nearChestCount = 0;
  let chestTouchCount = 0;
  let inwardPointCount = 0;
  const chestSamples = [];
  const chestDistances = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const handCenter = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? null;
    const indexDip = landmarks?.[7] ?? indexTip;
    const indexPip = landmarks?.[6] ?? indexDip;
    const handAnchor =
      indexTip && indexDip
        ? {
          x:
            Number(indexTip.x ?? 0) * 0.55 +
            Number(indexDip.x ?? 0) * 0.3 +
            Number(indexPip?.x ?? indexDip.x ?? 0) * 0.15,
          y:
            Number(indexTip.y ?? 0) * 0.55 +
            Number(indexDip.y ?? 0) * 0.3 +
            Number(indexPip?.y ?? indexDip.y ?? 0) * 0.15,
        }
        : (indexTip ?? handCenter);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const pointingGesture =
      (
        prediction?.gesture === "Указательный" &&
        Number(prediction?.confidence ?? 0) >= 0.28
      ) ||
      (
        Number(features?.openScores?.index ?? 0) >= 0.56 &&
        Number(features?.openScores?.middle ?? 1) <= 0.54 &&
        Number(features?.openScores?.ring ?? 1) <= 0.52 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.52
      );

    if (pointingGesture) {
      pointingCount += 1;
    }

    if (!handAnchor || !handCenter || !noSecondHand) {
      continue;
    }

    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const chestTarget =
      getChestTargetPoint(frame?.pose, headStats) ??
      (headStats
        ? {
            x: Number(headStats.centerX ?? 0.5),
            y: Number(headStats.eyeY ?? 0.34) + Number(headStats.scale ?? 0.1) * 1.35,
            scale: Math.max(Number(headStats.scale ?? 0.1), 0.08),
          }
        : null);

    if (!chestTarget) {
      continue;
    }

    const scale = Math.max(Number(chestTarget?.scale ?? 0.1), 0.08);
    const distanceToChest = getPointDistance(handAnchor, chestTarget) / scale;
    const handCenterDistance = getPointDistance(handCenter, chestTarget) / scale;
    const centerOffsetNorm =
      Math.abs(Number(handAnchor.x ?? 0) - Number(chestTarget.x ?? 0.5)) / scale;
    const centeredToChest = centerOffsetNorm <= 0.84;
    const belowFaceLine =
      !headStats || Number(handAnchor.y ?? 1) >= Number(headStats.eyeY ?? 0.34) + 0.01;
    const touchChest = distanceToChest <= 0.64 && centeredToChest && belowFaceLine;
    const nearChest = distanceToChest <= 0.98 && centeredToChest && belowFaceLine;
    const inwardPointing = distanceToChest <= handCenterDistance + 0.24;

    if (nearChest) {
      nearChestCount += 1;
      if (inwardPointing) {
        inwardPointCount += 1;
      }
      if (touchChest) {
        chestTouchCount += 1;
      }
      chestSamples.push({
        x: Number(handAnchor.x ?? 0),
        y: Number(handAnchor.y ?? 0),
        scale,
        centerOffsetNorm,
      });
      chestDistances.push(distanceToChest);
    }
  }

  if (chestSamples.length < 4) {
    return null;
  }

  const pointingRatio = pointingCount / handFrames.length;
  const nearChestRatio = nearChestCount / handFrames.length;
  const chestTouchRatio = chestTouchCount / handFrames.length;
  const inwardPointRatio =
    nearChestCount > 0 ? inwardPointCount / nearChestCount : 0;
  const scaleAverage = Math.max(
    average(chestSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = chestSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = chestSamples.map((sample) => Number(sample.y ?? 0));
  const centerOffsets = chestSamples.map((sample) => Number(sample.centerOffsetNorm ?? 1));
  const distanceToChestAverage = average(chestDistances);
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const centerOffsetAverage = average(centerOffsets);
  const stableSelfPoint = xRangeNorm <= 0.48 && yRangeNorm <= 0.56;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.42) / 0.5) * 0.18 +
    clamp01((pointingRatio - 0.16) / 0.7) * 0.22 +
    clamp01((nearChestRatio - 0.14) / 0.68) * 0.24 +
    clamp01((chestTouchRatio - 0.1) / 0.42) * 0.18 +
    clamp01((inwardPointRatio - 0.4) / 0.56) * 0.22 +
    clamp01((0.62 - centerOffsetAverage) / 0.62) * 0.16 +
    clamp01((0.56 - xRangeNorm) / 0.56) * 0.07 +
    clamp01((0.62 - yRangeNorm) / 0.62) * 0.05 +
    (stableSelfPoint ? 0.02 : 0),
  );
  const strictGate =
    oneHandRatio >= 0.42 &&
    pointingRatio >= 0.14 &&
    nearChestRatio >= 0.14 &&
    chestTouchRatio >= 0.1 &&
    inwardPointRatio >= 0.38 &&
    centerOffsetAverage <= 0.78 &&
    (stableSelfPoint || chestTouchRatio >= 0.14 || nearChestRatio >= 0.24);

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Я" ||
    current.confidence < 0.78 ||
    (
      ["Ты", "Привет", "Мужчина", "Женщина", "Стоп"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Я" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Я" ? current.label : "Ты";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Я",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-ya-self-point-detector",
    scores: [
      { label: "Я", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "ya_point_to_self_chest",
      detectorScore,
      oneHandRatio,
      pointingRatio,
      nearChestRatio,
      chestTouchRatio,
      inwardPointRatio,
      centerOffsetAverage,
      distanceToChestAverage,
      xRangeNorm,
      yRangeNorm,
      stableSelfPoint,
      dominantHand,
    },
  };
}

function buildTyPointYouPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let pointingCount = 0;
  let outwardCount = 0;
  let selfLikeCount = 0;
  const outwardSamples = [];
  const outwardMargins = [];
  const outwardDistances = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const handCenter = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? handCenter;
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const pointingGesture =
      (
        prediction?.gesture === "Указательный" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(features?.openScores?.index ?? 0) >= 0.62 &&
        Number(features?.openScores?.middle ?? 1) <= 0.48 &&
        Number(features?.openScores?.ring ?? 1) <= 0.46 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.46
      );

    if (pointingGesture) {
      pointingCount += 1;
    }

    if (!pointingGesture || !indexTip || !handCenter || !noSecondHand) {
      continue;
    }

    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const chestTarget = getChestTargetPoint(frame?.pose, headStats);

    if (!chestTarget) {
      continue;
    }

    const scale = Math.max(Number(chestTarget?.scale ?? 0.1), 0.08);
    const distanceToChest = getPointDistance(indexTip, chestTarget) / scale;
    const handCenterDistance = getPointDistance(handCenter, chestTarget) / scale;
    const distanceToHeadCenter = headStats
      ? getPointDistance(indexTip, {
        x: Number(headStats.centerX ?? 0.5),
        y: Number(headStats.eyeY ?? 0.34),
      }) / Math.max(Number(headStats.scale ?? scale), 0.08)
      : 1.5;
    const outwardPointing =
      distanceToChest >= 0.96 &&
      distanceToChest >= handCenterDistance + 0.08 &&
      distanceToHeadCenter >= 0.62;
    const selfLikePoint = distanceToChest <= 1.02;

    if (selfLikePoint) {
      selfLikeCount += 1;
    }

    if (outwardPointing) {
      outwardCount += 1;
      outwardMargins.push(distanceToChest - handCenterDistance);
      outwardDistances.push(distanceToChest);
      outwardSamples.push({
        x: Number(indexTip.x ?? 0),
        y: Number(indexTip.y ?? 0),
        scale,
      });
    }
  }

  if (outwardSamples.length < 4) {
    return null;
  }

  const pointingRatio = pointingCount / handFrames.length;
  const outwardRatio = outwardCount / handFrames.length;
  const selfLikeRatio = selfLikeCount / handFrames.length;
  const outwardMarginAverage = average(outwardMargins);
  const distanceToChestAverage = average(outwardDistances);
  const scaleAverage = Math.max(
    average(outwardSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = outwardSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = outwardSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const stableDirection = xRangeNorm <= 0.64 && yRangeNorm <= 0.54;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.42) * 0.18 +
    clamp01((pointingRatio - 0.24) / 0.68) * 0.22 +
    clamp01((outwardRatio - 0.22) / 0.62) * 0.26 +
    clamp01((0.52 - selfLikeRatio) / 0.52) * 0.14 +
    clamp01((0.6 - xRangeNorm) / 0.6) * 0.08 +
    clamp01((0.5 - yRangeNorm) / 0.5) * 0.07 +
    (stableDirection ? 0.03 : 0),
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    pointingRatio >= 0.34 &&
    outwardRatio >= 0.28 &&
    selfLikeRatio <= 0.4 &&
    distanceToChestAverage >= 1.08 &&
    outwardMarginAverage >= 0.12 &&
    stableDirection;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Ты" ||
    current.confidence < 0.78 ||
    (
      ["Я", "Привет", "Нет", "Да", "Стоп"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Ты" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Ты" ? current.label : "Я";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Ты",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-ty-point-you-detector",
    scores: [
      { label: "Ты", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "ty_point_to_person",
      detectorScore,
      oneHandRatio,
      pointingRatio,
      outwardRatio,
      selfLikeRatio,
      outwardMarginAverage,
      distanceToChestAverage,
      xRangeNorm,
      yRangeNorm,
      stableDirection,
      dominantHand,
    },
  };
}

function buildHoroshoThumbUpPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let thumbUpCount = 0;
  let chestLevelCount = 0;
  let poseSupportedFrames = 0;
  const thumbSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const thumbUpDetected =
      (
        prediction?.gesture === "Палец вверх" &&
        Number(prediction?.confidence ?? 0) >= 0.36
      ) ||
      (
        Number(features?.openScores?.thumb ?? 0) >= 0.62 &&
        Number(features?.openScores?.index ?? 1) <= 0.48 &&
        Number(features?.openScores?.middle ?? 1) <= 0.48 &&
        Number(features?.openScores?.ring ?? 1) <= 0.48 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.48
      );

    if (thumbUpDetected) {
      thumbUpCount += 1;
    }

    if (!thumbUpDetected || !center || !noSecondHand) {
      continue;
    }

    if (shoulderStats) {
      poseSupportedFrames += 1;
      const minChestY = shoulderStats.centerY - 0.22;
      const maxChestY = shoulderStats.centerY + 0.4;
      const isChestLevel =
        Number(center.y ?? 0) >= minChestY &&
        Number(center.y ?? 0) <= maxChestY;

      if (isChestLevel) {
        chestLevelCount += 1;
      }
    }

    thumbSamples.push({
      x: Number(center.x ?? 0),
      y: Number(center.y ?? 0),
      scale: Number(shoulderStats?.scale ?? 0.12),
    });
  }

  if (thumbSamples.length < 4) {
    return null;
  }

  const thumbUpRatio = thumbUpCount / handFrames.length;
  const chestRatio =
    poseSupportedFrames > 0 ? chestLevelCount / poseSupportedFrames : 1;
  const scaleAverage = Math.max(
    average(thumbSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = thumbSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = thumbSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const stableThumbHold = xRangeNorm <= 0.46 && yRangeNorm <= 0.58;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.48) / 0.46) * 0.2 +
    clamp01((thumbUpRatio - 0.26) / 0.7) * 0.34 +
    clamp01((chestRatio - 0.28) / 0.68) * 0.2 +
    clamp01((0.5 - xRangeNorm) / 0.5) * 0.1 +
    clamp01((0.62 - yRangeNorm) / 0.62) * 0.12 +
    (stableThumbHold ? 0.04 : 0),
  );
  const strictGate =
    oneHandRatio >= 0.52 &&
    thumbUpRatio >= 0.32 &&
    (poseSupportedFrames < 3 || chestRatio >= 0.34) &&
    stableThumbHold;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Хорошо" ||
    current.confidence < 0.82 ||
    (
      ["Да", "Нет", "Стоп", "Ты", "Я"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Хорошо" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Хорошо" ? current.label : "Да";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.77 - detectorScore * 0.16),
  );

  return {
    label: "Хорошо",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-horosho-thumb-up-detector",
    scores: [
      { label: "Хорошо", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "horosho_thumb_up",
      detectorScore,
      oneHandRatio,
      thumbUpRatio,
      chestRatio,
      xRangeNorm,
      yRangeNorm,
      stableThumbHold,
      dominantHand,
    },
  };
}

function buildStopOpenPalmPrediction({
  frames = [],
  metadata = null,
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let openPalmCount = 0;
  let stopPoseCount = 0;
  let nearHeadCount = 0;
  const stopSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const openPalmDetected =
      (
        prediction?.gesture === "Открытая ладонь" &&
        Number(prediction?.confidence ?? 0) >= 0.42
      ) ||
      (
        Number(features?.longFingerRelaxedCount ?? 0) >= 3 &&
        Number(features?.fingerExtensionAverage ?? 0) >= 0.6
      );

    if (openPalmDetected) {
      openPalmCount += 1;
    }

    if (!openPalmDetected || !center || !noSecondHand) {
      continue;
    }

    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const chestTarget = getChestTargetPoint(frame?.pose, headStats);
    const scale = Math.max(
      Number(chestTarget?.scale ?? shoulderStats?.scale ?? headStats?.scale ?? 0.12),
      0.08,
    );
    const minY = Number(headStats?.eyeY ?? 0.3) - 0.08;
    const maxY = Number(shoulderStats?.centerY ?? chestTarget?.y ?? 0.56) + 0.42;
    const withinBodyZone =
      Number(center.y ?? 0) >= minY &&
      Number(center.y ?? 0) <= maxY;
    const notAboveHead =
      !headStats || Number(center.y ?? 1) >= Number(headStats.topY ?? 0.22) - 0.04;
    const distanceToHeadCenter = headStats
      ? getPointDistance(center, {
        x: Number(headStats.centerX ?? 0.5),
        y: Number(headStats.eyeY ?? 0.34),
      }) / Math.max(Number(headStats.scale ?? scale), 0.08)
      : 1.2;

    if (distanceToHeadCenter <= 0.9) {
      nearHeadCount += 1;
    }

    const stopPose = withinBodyZone && notAboveHead && distanceToHeadCenter >= 0.54;

    if (stopPose) {
      stopPoseCount += 1;
      stopSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
        scale,
      });
    }
  }

  if (stopSamples.length < 4) {
    return null;
  }

  const waveScore = Number(metadata?.waveScore ?? 0);
  const openPalmRatio = openPalmCount / handFrames.length;
  const stopPoseRatio = stopPoseCount / handFrames.length;
  const nearHeadRatio = nearHeadCount / handFrames.length;
  const scaleAverage = Math.max(
    average(stopSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = stopSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = stopSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const xDirectionChanges = countDirectionChanges(xValues, 0.0015);
  const stableStopHold =
    xRangeNorm <= 0.44 &&
    yRangeNorm <= 0.42 &&
    xDirectionChanges <= 2;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.42) * 0.2 +
    clamp01((openPalmRatio - 0.34) / 0.62) * 0.26 +
    clamp01((stopPoseRatio - 0.24) / 0.68) * 0.22 +
    clamp01((0.62 - nearHeadRatio) / 0.62) * 0.09 +
    clamp01((0.48 - xRangeNorm) / 0.48) * 0.09 +
    clamp01((0.46 - yRangeNorm) / 0.46) * 0.08 +
    clamp01((0.76 - waveScore) / 0.76) * 0.1 +
    (stableStopHold ? 0.04 : 0),
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    openPalmRatio >= 0.42 &&
    stopPoseRatio >= 0.28 &&
    nearHeadRatio <= 0.58 &&
    stableStopHold &&
    waveScore < 0.76;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Стоп" ||
    current.confidence < 0.82 ||
    (
      ["Пока", "Привет", "Солнце", "Хорошо", "Нет"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Стоп" ? current.confidence : 0,
    0.69 + detectorScore * 0.25,
  ));
  const secondaryLabel =
    current.label && current.label !== "Стоп" ? current.label : "Пока";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.77 - detectorScore * 0.16),
  );

  return {
    label: "Стоп",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-stop-open-palm-detector",
    scores: [
      { label: "Стоп", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "stop_open_palm_forward_hold",
      detectorScore,
      oneHandRatio,
      openPalmRatio,
      stopPoseRatio,
      nearHeadRatio,
      xRangeNorm,
      yRangeNorm,
      xDirectionChanges,
      stableStopHold,
      waveScore,
      dominantHand,
    },
  };
}

function buildEstEatPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.44))) {
    return null;
  }

  let nearMouthCount = 0;
  let pinchCount = 0;
  const distanceSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const mouthTarget = getMouthTarget(frame?.face, frame?.pose);

    if (!mouthTarget) {
      continue;
    }

    const indexTip = landmarks?.[8] ?? null;
    const thumbTip = landmarks?.[4] ?? null;
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!indexTip || !thumbTip || !noSecondHand) {
      continue;
    }

    const features = extractHandFeatures(landmarks, handedness);
    const pinch =
      getPointDistance(indexTip, thumbTip) / Math.max(Number(mouthTarget.scale ?? 0.1), 0.08) <= 0.6 ||
      Number(features?.thumbToIndexTip ?? 1) <= 0.54 ||
      Number(features?.fingerPinchLike ?? 0) >= 0.16;
    const mouthDistance =
      getPointDistance(indexTip, mouthTarget) / Math.max(Number(mouthTarget.scale ?? 0.1), 0.08);
    const nearMouth = mouthDistance <= 1.14;

    if (pinch) {
      pinchCount += 1;
    }

    if (nearMouth) {
      nearMouthCount += 1;
    }

    if (pinch && nearMouth) {
      distanceSamples.push(mouthDistance);
    }
  }

  if (distanceSamples.length < 4) {
    return null;
  }

  const pinchRatio = pinchCount / handFrames.length;
  const nearMouthRatio = nearMouthCount / handFrames.length;
  const mouthDirectionChanges = countDirectionChanges(distanceSamples, 0.02);
  const hasCycles = mouthDirectionChanges >= 1;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.36) / 0.54) * 0.2 +
    clamp01((pinchRatio - 0.18) / 0.68) * 0.3 +
    clamp01((nearMouthRatio - 0.18) / 0.68) * 0.24 +
    clamp01(mouthDirectionChanges / 3) * 0.26,
  );
  const strictGate =
    oneHandRatio >= 0.4 &&
    pinchRatio >= 0.2 &&
    nearMouthRatio >= 0.2 &&
    hasCycles;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Есть" ||
    current.confidence < 0.84 ||
    (
      ["Пить"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Есть" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Есть" ? current.label : "Пить";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Есть",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-est-eat-detector",
    scores: [
      { label: "Есть", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "est_fingers_to_mouth",
      detectorScore,
      oneHandRatio,
      pinchRatio,
      nearMouthRatio,
      mouthDirectionChanges,
      dominantHand,
    },
  };
}

function buildPitDrinkPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.44))) {
    return null;
  }

  let cupShapeCount = 0;
  let nearMouthCount = 0;
  const mouthDistanceSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const mouthTarget = getMouthTarget(frame?.face, frame?.pose);
    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!mouthTarget || !noSecondHand) {
      continue;
    }

    const center = getHandCenter(landmarks);
    const features = extractHandFeatures(landmarks, handedness);
    const mouthPoint = mouthTarget;

    if (!center || !features || !mouthPoint) {
      continue;
    }

    const curledEnough = Number(features.curledLongFingerCount ?? 0) >= 1;
    const notFist = Number(features.openCount ?? 0) >= 1;
    const cupShape =
      curledEnough &&
      notFist &&
      Number(features.fingerExtensionAverage ?? 0) <= 0.76 &&
      Number(features.thumbToIndexTip ?? 1) >= 0.26;
    const mouthDistance =
      getPointDistance(center, mouthPoint) / Math.max(Number(mouthPoint.scale ?? 0.1), 0.08);
    const nearMouth = mouthDistance <= 1.12;
    const underNose = !headStats || Number(center.y ?? 0.5) >= Number(headStats.eyeY ?? 0.34) - 0.03;

    if (cupShape) {
      cupShapeCount += 1;
    }

    if (nearMouth && underNose) {
      nearMouthCount += 1;
    }

    if (cupShape) {
      mouthDistanceSamples.push(mouthDistance);
    }
  }

  if (mouthDistanceSamples.length < 4) {
    return null;
  }

  const cupShapeRatio = cupShapeCount / handFrames.length;
  const nearMouthRatio = nearMouthCount / handFrames.length;
  const startDistance = average(mouthDistanceSamples.slice(0, Math.max(2, Math.round(mouthDistanceSamples.length * 0.35))));
  const endDistance = average(mouthDistanceSamples.slice(-Math.max(2, Math.round(mouthDistanceSamples.length * 0.35))));
  const towardMouthDelta = startDistance - endDistance;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.4) * 0.2 +
    clamp01((cupShapeRatio - 0.28) / 0.68) * 0.32 +
    clamp01((nearMouthRatio - 0.2) / 0.72) * 0.26 +
    clamp01((towardMouthDelta - 0.04) / 0.6) * 0.22,
  );
  const strictGate =
    oneHandRatio >= 0.46 &&
    cupShapeRatio >= 0.24 &&
    nearMouthRatio >= 0.22 &&
    towardMouthDelta >= 0.04;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Пить" ||
    current.confidence < 0.84 ||
    (
      ["Есть"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Пить" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Пить" ? current.label : "Есть";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Пить",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-pit-drink-detector",
    scores: [
      { label: "Пить", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "pit_cup_to_mouth",
      detectorScore,
      oneHandRatio,
      cupShapeRatio,
      nearMouthRatio,
      towardMouthDelta,
      dominantHand,
    },
  };
}

function buildSpatSleepPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  let palmCount = 0;
  let nearChinCount = 0;
  let closedEyesCount = 0;
  let pinchEndCount = 0;
  const ySamples = [];
  const scaleSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const prediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const face = Array.isArray(frame?.face) ? frame.face : [];
    const chinTarget =
      face?.[152]
        ? {
            x: Number(face[152].x ?? 0.5),
            y: Number(face[152].y ?? 0.6),
            scale: Math.max(Number(headStats?.scale ?? 0.1), 0.08),
          }
        : null;
    const center = getHandCenter(landmarks);
    const thumbTip = landmarks?.[4] ?? null;
    const indexTip = landmarks?.[8] ?? null;
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!chinTarget || !headStats || !center || !noSecondHand) {
      continue;
    }

    const openPalm =
      (
        prediction?.gesture === "Открытая ладонь" &&
        Number(prediction?.confidence ?? 0) >= 0.24
      ) ||
      Number(features?.longFingerRelaxedCount ?? 0) >= 2 ||
      Number(features?.fingerExtensionAverage ?? 0) >= 0.4;
    const chinDistance =
      getPointDistance(center, chinTarget) / Math.max(Number(chinTarget.scale ?? 0.1), 0.08);
    const nearChin = chinDistance <= 1.02;
    const leftEyeTop = face?.[159] ?? null;
    const leftEyeBottom = face?.[145] ?? null;
    const rightEyeTop = face?.[386] ?? null;
    const rightEyeBottom = face?.[374] ?? null;
    const leftEyeClosed =
      leftEyeTop &&
      leftEyeBottom &&
      Math.abs(Number(leftEyeTop.y ?? 0.4) - Number(leftEyeBottom.y ?? 0.4)) /
        Math.max(Number(headStats.scale ?? 0.1), 0.08) <= 0.18;
    const rightEyeClosed =
      rightEyeTop &&
      rightEyeBottom &&
      Math.abs(Number(rightEyeTop.y ?? 0.4) - Number(rightEyeBottom.y ?? 0.4)) /
        Math.max(Number(headStats.scale ?? 0.1), 0.08) <= 0.18;
    const eyesClosed = Boolean(leftEyeClosed && rightEyeClosed);
    const pinchEnd =
      thumbTip &&
      indexTip &&
      getPointDistance(thumbTip, indexTip) / Math.max(Number(headStats.scale ?? 0.1), 0.08) <= 0.56;

    if (openPalm) {
      palmCount += 1;
    }

    if (nearChin) {
      nearChinCount += 1;
    }
    if (eyesClosed) {
      closedEyesCount += 1;
    }
    if (nearChin && pinchEnd) {
      pinchEndCount += 1;
    }
    if (openPalm) {
      ySamples.push(Number(center.y ?? 0));
      scaleSamples.push(Math.max(Number(headStats.scale ?? 0.1), 0.08));
    }
  }

  if (ySamples.length < 4) {
    return null;
  }

  const scaleAverage = Math.max(average(scaleSamples), 0.08);
  const palmRatio = palmCount / handFrames.length;
  const nearChinRatio = nearChinCount / handFrames.length;
  const closedEyesRatio = closedEyesCount / handFrames.length;
  const pinchEndRatio = pinchEndCount / Math.max(nearChinCount, 1);
  const yRangeNorm =
    (Math.max(...ySamples) - Math.min(...ySamples)) / scaleAverage;
  const downwardSweep = yRangeNorm >= 0.16;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.36) / 0.58) * 0.18 +
    clamp01((palmRatio - 0.18) / 0.76) * 0.2 +
    clamp01((nearChinRatio - 0.18) / 0.76) * 0.24 +
    clamp01((closedEyesRatio - 0.12) / 0.78) * 0.2 +
    clamp01((pinchEndRatio - 0.08) / 0.82) * 0.08 +
    clamp01((yRangeNorm - 0.14) / 0.94) * 0.1,
  );
  const strictGate =
    oneHandRatio >= 0.3 &&
    palmRatio >= 0.18 &&
    nearChinRatio >= 0.18 &&
    downwardSweep &&
    (closedEyesRatio >= 0.12 || pinchEndRatio >= 0.12);

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Спать" ||
    current.confidence < 0.84 ||
    (
      ["Думать", "Женщина"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Спать" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Спать" ? current.label : "Женщина";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Спать",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-spat-sleep-detector",
    scores: [
      { label: "Спать", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "spat_palm_down_to_chin_eyes_closed",
      detectorScore,
      oneHandRatio,
      palmRatio,
      nearChinRatio,
      closedEyesRatio,
      pinchEndRatio,
      yRangeNorm,
      downwardSweep,
      dominantHand,
    },
  };
}

function buildIdtiWalkPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  let twoFingerCount = 0;
  const stepSamples = [];
  const scaleSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const center = getHandCenter(landmarks);
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const prediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(Number(shoulderStats?.scale ?? 0.12), 0.08);
    const twoFingers =
      (
        prediction?.gesture === "Два пальца" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(features?.openScores?.index ?? 0) >= 0.62 &&
        Number(features?.openScores?.middle ?? 0) >= 0.58 &&
        Number(features?.openScores?.ring ?? 1) <= 0.48 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.48
      );

    if (twoFingers) {
      twoFingerCount += 1;
    }

    if (twoFingers && center && noSecondHand) {
      stepSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
      });
      scaleSamples.push(frameScale);
    }
  }

  if (stepSamples.length < 4) {
    return null;
  }

  const scaleAverage = Math.max(average(scaleSamples), 0.08);
  const xValues = stepSamples.map((sample) => sample.x);
  const yValues = stepSamples.map((sample) => sample.y);
  const xRangeNorm = (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm = (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const directionChanges = countDirectionChanges(xValues, 0.0022);
  const speedScore =
    average(xValues.slice(1).map((value, index) => Math.abs(value - xValues[index]))) / 0.01;
  const twoFingerRatio = twoFingerCount / handFrames.length;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.4) * 0.2 +
    clamp01((twoFingerRatio - 0.28) / 0.68) * 0.34 +
    clamp01((xRangeNorm - 0.06) / 0.5) * 0.22 +
    clamp01(directionChanges / 3) * 0.14 +
    clamp01((0.32 - speedScore) / 0.32) * 0.1,
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    twoFingerRatio >= 0.34 &&
    xRangeNorm >= 0.08 &&
    xRangeNorm <= 0.5 &&
    yRangeNorm <= 0.42 &&
    directionChanges >= 1 &&
    speedScore >= 0.08 &&
    speedScore <= 0.28;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Идти" ||
    current.confidence < 0.84 ||
    (
      ["Бежать", "Ты"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Идти" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Идти" ? current.label : "Бежать";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Идти",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-idti-walk-detector",
    scores: [
      { label: "Идти", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "idti_two_finger_walk",
      detectorScore,
      oneHandRatio,
      twoFingerRatio,
      xRangeNorm,
      yRangeNorm,
      directionChanges,
      speedScore,
      dominantHand,
    },
  };
}

function buildBezhatRunPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  let twoFingerCount = 0;
  const stepSamples = [];
  const scaleSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const center = getHandCenter(landmarks);
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const prediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(Number(shoulderStats?.scale ?? 0.12), 0.08);
    const twoFingers =
      (
        prediction?.gesture === "Два пальца" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(features?.openScores?.index ?? 0) >= 0.62 &&
        Number(features?.openScores?.middle ?? 0) >= 0.58 &&
        Number(features?.openScores?.ring ?? 1) <= 0.48 &&
        Number(features?.openScores?.pinky ?? 1) <= 0.48
      );

    if (twoFingers) {
      twoFingerCount += 1;
    }

    if (twoFingers && center && noSecondHand) {
      stepSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
      });
      scaleSamples.push(frameScale);
    }
  }

  if (stepSamples.length < 4) {
    return null;
  }

  const scaleAverage = Math.max(average(scaleSamples), 0.08);
  const xValues = stepSamples.map((sample) => sample.x);
  const yValues = stepSamples.map((sample) => sample.y);
  const xRangeNorm = (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm = (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const directionChanges = countDirectionChanges(xValues, 0.0024);
  const speedScore =
    average(xValues.slice(1).map((value, index) => Math.abs(value - xValues[index]))) / 0.01;
  const twoFingerRatio = twoFingerCount / handFrames.length;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.4) * 0.2 +
    clamp01((twoFingerRatio - 0.28) / 0.68) * 0.3 +
    clamp01((xRangeNorm - 0.1) / 0.62) * 0.22 +
    clamp01(directionChanges / 4) * 0.14 +
    clamp01((speedScore - 0.2) / 0.46) * 0.14,
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    twoFingerRatio >= 0.34 &&
    xRangeNorm >= 0.12 &&
    xRangeNorm <= 0.72 &&
    yRangeNorm <= 0.54 &&
    directionChanges >= 2 &&
    speedScore >= 0.2;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Бежать" ||
    current.confidence < 0.84 ||
    (
      ["Идти", "Ты"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Бежать" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Бежать" ? current.label : "Идти";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Бежать",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-bezhat-run-detector",
    scores: [
      { label: "Бежать", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "bezhat_two_finger_run",
      detectorScore,
      oneHandRatio,
      twoFingerRatio,
      xRangeNorm,
      yRangeNorm,
      directionChanges,
      speedScore,
      dominantHand,
    },
  };
}

function buildDumatThinkPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  let indexCount = 0;
  let nearTempleCount = 0;

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const indexTip = landmarks?.[8] ?? null;
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const prediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const eyePoint = getFaceFeaturePoint(
      frame?.face,
      dominantHand === "left" ? [33, 133, 159, 145] : [362, 263, 386, 374],
    );

    if (!indexTip || !noSecondHand || !headStats || !eyePoint) {
      continue;
    }

    const indexGesture =
      (
        prediction?.gesture === "Указательный" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      Number(features?.openScores?.index ?? 0) >= 0.62;
    const templePoint = {
      x:
        Number(eyePoint.x ?? 0.5) +
        (dominantHand === "left" ? -1 : 1) * Number(headStats.scale ?? 0.1) * 0.22,
      y: Number(eyePoint.y ?? 0.35) - Number(headStats.scale ?? 0.1) * 0.02,
    };
    const templeDistance =
      getPointDistance(indexTip, templePoint) / Math.max(Number(headStats.scale ?? 0.1), 0.08);

    if (indexGesture) {
      indexCount += 1;
      if (templeDistance <= 1.08) {
        nearTempleCount += 1;
      }
    }
  }

  const indexRatio = indexCount / handFrames.length;
  const nearTempleRatio = nearTempleCount / handFrames.length;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.4) * 0.2 +
    clamp01((indexRatio - 0.3) / 0.66) * 0.36 +
    clamp01((nearTempleRatio - 0.24) / 0.72) * 0.44,
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    indexRatio >= 0.34 &&
    nearTempleRatio >= 0.28;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Думать" ||
    current.confidence < 0.84 ||
    (
      ["Мужчина"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Думать" ? current.confidence : 0,
    0.68 + detectorScore * 0.26,
  ));
  const secondaryLabel =
    current.label && current.label !== "Думать" ? current.label : "Женщина";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Думать",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-dumat-think-detector",
    scores: [
      { label: "Думать", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "dumat_finger_to_temple",
      detectorScore,
      oneHandRatio,
      indexRatio,
      nearTempleRatio,
      dominantHand,
    },
  };
}

function buildLyubovLovePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });

  if (pairedFrames.length < Math.max(6, Math.round(frames.length * 0.38))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let chestCrossCount = 0;
  let holdCount = 0;
  let symmetricCount = 0;
  let elbowsDownCount = 0;
  let chestLevelCount = 0;
  const crossXValues = [];

  for (const frame of pairedFrames) {
    const leftCenter = getHandCenter(frame?.left_hand);
    const rightCenter = getHandCenter(frame?.right_hand);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const chestTarget = getChestTargetPoint(frame?.pose, getHeadFrameStats(frame?.face, frame?.pose));
    const leftElbow = frame?.pose?.[13] ?? null;
    const rightElbow = frame?.pose?.[14] ?? null;
    const leftShoulder = frame?.pose?.[11] ?? null;
    const rightShoulder = frame?.pose?.[12] ?? null;

    if (!leftCenter || !rightCenter || !chestTarget) {
      continue;
    }

    const scale = Math.max(Number(shoulderStats?.scale ?? chestTarget.scale ?? 0.12), 0.08);
    const leftToChest = getPointDistance(leftCenter, chestTarget) / scale;
    const rightToChest = getPointDistance(rightCenter, chestTarget) / scale;
    const crossed =
      Number(leftCenter.x ?? 0) > Number(chestTarget.x ?? 0.5) &&
      Number(rightCenter.x ?? 0) < Number(chestTarget.x ?? 0.5);
    const nearChest = leftToChest <= 1.24 && rightToChest <= 1.24;
    const chestLevel =
      Math.abs(Number(leftCenter.y ?? 0) - Number(chestTarget.y ?? 0.5)) / scale <= 0.9 &&
      Math.abs(Number(rightCenter.y ?? 0) - Number(chestTarget.y ?? 0.5)) / scale <= 0.9;
    const closeHands = getPointDistance(leftCenter, rightCenter) / scale <= 1.64;
    const leftRelX = (Number(leftCenter.x ?? 0.5) - Number(chestTarget.x ?? 0.5)) / scale;
    const rightRelX = (Number(chestTarget.x ?? 0.5) - Number(rightCenter.x ?? 0.5)) / scale;
    const leftRelY = (Number(leftCenter.y ?? 0.5) - Number(chestTarget.y ?? 0.5)) / scale;
    const rightRelY = (Number(rightCenter.y ?? 0.5) - Number(chestTarget.y ?? 0.5)) / scale;
    const symmetric =
      Math.abs(leftRelX - rightRelX) <= 0.62 &&
      Math.abs(leftRelY - rightRelY) <= 0.62;
    const elbowsDetected = Boolean(
      leftElbow &&
      rightElbow &&
      leftShoulder &&
      rightShoulder,
    );
    const elbowsDownAndOut = elbowsDetected
      ? (
        Number(leftElbow.y ?? 0) >= Number(leftShoulder.y ?? 0) + 0.02 &&
        Number(rightElbow.y ?? 0) >= Number(rightShoulder.y ?? 0) + 0.02 &&
        Number(leftElbow.x ?? 0.5) <= Number(leftShoulder.x ?? 0.5) + 0.04 &&
        Number(rightElbow.x ?? 0.5) >= Number(rightShoulder.x ?? 0.5) - 0.04
      )
      : false;

    if (crossed && nearChest && closeHands && chestLevel) {
      chestCrossCount += 1;
      if (chestLevel) {
        chestLevelCount += 1;
      }
      if (symmetric) {
        symmetricCount += 1;
      }
      if (elbowsDownAndOut) {
        elbowsDownCount += 1;
      }
      crossXValues.push(Number(leftCenter.x ?? 0) - Number(rightCenter.x ?? 0));
    }
  }

  if (crossXValues.length < 4) {
    return null;
  }

  const crossRatio = chestCrossCount / pairedFrames.length;
  const symmetricRatio = symmetricCount / pairedFrames.length;
  const elbowsDownRatio = elbowsDownCount / pairedFrames.length;
  const chestLevelRatio = chestLevelCount / pairedFrames.length;
  const crossRange = Math.max(...crossXValues) - Math.min(...crossXValues);
  const stableHold = crossRange <= 0.16;
  holdCount = stableHold ? Math.max(1, Math.round(crossRatio * pairedFrames.length)) : 0;
  const holdRatio = holdCount / pairedFrames.length;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.36) / 0.58) * 0.2 +
    clamp01((crossRatio - 0.28) / 0.68) * 0.26 +
    clamp01((symmetricRatio - 0.24) / 0.72) * 0.2 +
    clamp01((chestLevelRatio - 0.26) / 0.72) * 0.14 +
    clamp01((elbowsDownRatio - 0.2) / 0.72) * 0.1 +
    clamp01((holdRatio - 0.42) / 0.5) * 0.06 +
    clamp01((0.18 - crossRange) / 0.18) * 0.04,
  );
  const strictGate =
    twoHandsRatio >= 0.44 &&
    crossRatio >= 0.34 &&
    symmetricRatio >= 0.24 &&
    chestLevelRatio >= 0.28 &&
    (elbowsDownRatio >= 0.16 || holdRatio >= 0.56) &&
    stableHold;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Любовь" ||
    current.confidence < 0.86 ||
    (
      ["Дружба", "Дом"].includes(current.label) &&
      current.confidence < 0.92
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Любовь" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Любовь" ? current.label : "Дружба";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.8 - detectorScore * 0.16),
  );

  return {
    label: "Любовь",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-lyubov-love-detector",
    scores: [
      { label: "Любовь", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "lyubov_cross_arms_chest",
      detectorScore,
      twoHandsRatio,
      crossRatio,
      symmetricRatio,
      elbowsDownRatio,
      chestLevelRatio,
      crossRange,
      holdCount,
      holdRatio,
    },
  };
}

function buildBolshoyWidePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });

  if (pairedFrames.length < Math.max(6, Math.round(frames.length * 0.38))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let openHandsCount = 0;
  let wideCount = 0;
  let symmetricCount = 0;
  let chestLevelCount = 0;
  let outsideShouldersCount = 0;
  const spreadSamples = [];

  for (const frame of pairedFrames) {
    const leftHand = frame?.left_hand;
    const rightHand = frame?.right_hand;
    const leftCenter = getHandCenter(leftHand);
    const rightCenter = getHandCenter(rightHand);
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const leftShoulder = frame?.pose?.[11] ?? null;
    const rightShoulder = frame?.pose?.[12] ?? null;

    if (!leftCenter || !rightCenter || !leftShoulder || !rightShoulder) {
      continue;
    }

    const scale = Math.max(Number(shoulderStats?.scale ?? 0.12), 0.08);
    const leftFeatures = extractHandFeatures(leftHand, "Left");
    const rightFeatures = extractHandFeatures(rightHand, "Right");
    const leftPrediction = predictGesture(leftHand, "Left");
    const rightPrediction = predictGesture(rightHand, "Right");
    const leftOpen =
      (
        leftPrediction?.gesture === "Открытая ладонь" &&
        Number(leftPrediction?.confidence ?? 0) >= 0.18
      ) ||
      Number(leftFeatures?.fingerExtensionAverage ?? 0) >= 0.34 ||
      Number(leftFeatures?.longFingerRelaxedCount ?? 0) >= 2;
    const rightOpen =
      (
        rightPrediction?.gesture === "Открытая ладонь" &&
        Number(rightPrediction?.confidence ?? 0) >= 0.18
      ) ||
      Number(rightFeatures?.fingerExtensionAverage ?? 0) >= 0.34 ||
      Number(rightFeatures?.longFingerRelaxedCount ?? 0) >= 2;

    if (leftOpen && rightOpen) {
      openHandsCount += 1;
    }

    const spreadNorm =
      Math.abs(Number(rightCenter.x ?? 0.5) - Number(leftCenter.x ?? 0.5)) / scale;
    const avgY = (Number(leftCenter.y ?? 0.5) + Number(rightCenter.y ?? 0.5)) / 2;
    const shoulderMidY =
      (Number(leftShoulder.y ?? 0.5) + Number(rightShoulder.y ?? 0.5)) / 2;
    const chestLevel = Math.abs(avgY - shoulderMidY) / scale <= 1.46;
    const symmetric =
      Math.abs(Number(leftCenter.y ?? 0.5) - Number(rightCenter.y ?? 0.5)) / scale <= 1.02;
    const outsideShoulders =
      Number(leftCenter.x ?? 0.5) <= Number(leftShoulder.x ?? 0.5) + 0.12 &&
      Number(rightCenter.x ?? 0.5) >= Number(rightShoulder.x ?? 0.5) - 0.12;
    const wide =
      leftOpen &&
      rightOpen &&
      spreadNorm >= 1.46 &&
      chestLevel &&
      symmetric;

    if (wide) {
      wideCount += 1;
      if (chestLevel) {
        chestLevelCount += 1;
      }
      if (symmetric) {
        symmetricCount += 1;
      }
      if (outsideShoulders) {
        outsideShouldersCount += 1;
      }
      spreadSamples.push(spreadNorm);
    }
  }

  if (spreadSamples.length < 3) {
    return null;
  }

  const openHandsRatio = openHandsCount / pairedFrames.length;
  const wideRatio = wideCount / pairedFrames.length;
  const symmetricRatio = symmetricCount / pairedFrames.length;
  const chestLevelRatio = chestLevelCount / pairedFrames.length;
  const outsideShouldersRatio = outsideShouldersCount / pairedFrames.length;
  const spreadAverage = average(spreadSamples);
  const spreadRange = Math.max(...spreadSamples) - Math.min(...spreadSamples);
  const stableHold = spreadRange <= 1.18;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.28) / 0.68) * 0.16 +
    clamp01((openHandsRatio - 0.22) / 0.72) * 0.18 +
    clamp01((wideRatio - 0.18) / 0.74) * 0.24 +
    clamp01((symmetricRatio - 0.16) / 0.78) * 0.12 +
    clamp01((chestLevelRatio - 0.18) / 0.76) * 0.1 +
    clamp01((outsideShouldersRatio - 0.08) / 0.8) * 0.08 +
    clamp01((spreadAverage - 1.34) / 1.28) * 0.16 +
    clamp01((1.2 - spreadRange) / 1.2) * 0.08
  );
  const strictGate =
    twoHandsRatio >= 0.32 &&
    openHandsRatio >= 0.22 &&
    wideRatio >= 0.18 &&
    symmetricRatio >= 0.14 &&
    chestLevelRatio >= 0.16 &&
    outsideShouldersRatio >= 0.08 &&
    spreadAverage >= 1.56 &&
    stableHold;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Большой" ||
    current.confidence < 0.82 ||
    (
      ["Дом", "Дружба", "Любовь", "Женщина", "Мужчина"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Большой" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Большой" ? current.label : "Дом";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Большой",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-bolshoy-wide-detector",
    scores: [
      { label: "Большой", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "bolshoy_wide_volume",
      detectorScore,
      twoHandsRatio,
      openHandsRatio,
      wideRatio,
      symmetricRatio,
      chestLevelRatio,
      outsideShouldersRatio,
      spreadAverage,
      spreadRange,
      stableHold,
    },
  };
}

function buildMalenkiySmallPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });
  const anyHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft || hasRight;
  });

  if (anyHandFrames.length < Math.max(6, Math.round(frames.length * 0.36))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let closeHandsCount = 0;
  let chestLevelCount = 0;
  let pinchCount = 0;
  let parallelCount = 0;
  const gapSamples = [];

  for (const frame of anyHandFrames) {
    const leftHand = Array.isArray(frame?.left_hand) ? frame.left_hand : null;
    const rightHand = Array.isArray(frame?.right_hand) ? frame.right_hand : null;
    const leftCenter = leftHand ? getHandCenter(leftHand) : null;
    const rightCenter = rightHand ? getHandCenter(rightHand) : null;
    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const leftShoulder = frame?.pose?.[11] ?? null;
    const rightShoulder = frame?.pose?.[12] ?? null;
    const scale = Math.max(Number(shoulderStats?.scale ?? 0.12), 0.08);

    for (const [hand, handedness] of [[leftHand, "Left"], [rightHand, "Right"]]) {
      if (!hand) {
        continue;
      }
      const thumbTip = hand?.[4] ?? null;
      const indexTip = hand?.[8] ?? null;
      const features = extractHandFeatures(hand, handedness);
      const pinchLike =
        (
          thumbTip &&
          indexTip &&
          getPointDistance(thumbTip, indexTip) / scale <= 0.48
        ) ||
        Number(features?.fingerPinchLike ?? 0) >= 0.24 ||
        (
          Number(features?.openScores?.index ?? 0) >= 0.28 &&
          Number(features?.fingerExtensionAverage ?? 0) <= 0.62
        );

      if (pinchLike) {
        pinchCount += 1;
      }
    }

    if (
      !leftCenter ||
      !rightCenter ||
      !leftShoulder ||
      !rightShoulder
    ) {
      continue;
    }

    const gapNorm =
      Math.abs(Number(rightCenter.x ?? 0.5) - Number(leftCenter.x ?? 0.5)) / scale;
    const verticalGap =
      Math.abs(Number(rightCenter.y ?? 0.5) - Number(leftCenter.y ?? 0.5)) / scale;
    const avgY = (Number(leftCenter.y ?? 0.5) + Number(rightCenter.y ?? 0.5)) / 2;
    const shoulderMidY =
      (Number(leftShoulder.y ?? 0.5) + Number(rightShoulder.y ?? 0.5)) / 2;
    const chestLevel = Math.abs(avgY - shoulderMidY) / scale <= 1.4;
    const parallel = verticalGap <= 0.84;
    const closeHands = gapNorm <= 0.96 && gapNorm >= 0.12;

    if (closeHands) {
      closeHandsCount += 1;
      if (chestLevel) {
        chestLevelCount += 1;
      }
      if (parallel) {
        parallelCount += 1;
      }
      gapSamples.push(gapNorm);
    }
  }

  if (gapSamples.length < 3 && pinchCount < Math.max(3, Math.round(frames.length * 0.16))) {
    return null;
  }

  const closeHandsRatio =
    pairedFrames.length > 0 ? closeHandsCount / Math.max(pairedFrames.length, 1) : 0;
  const chestLevelRatio =
    pairedFrames.length > 0 ? chestLevelCount / Math.max(pairedFrames.length, 1) : 0;
  const parallelRatio =
    pairedFrames.length > 0 ? parallelCount / Math.max(pairedFrames.length, 1) : 0;
  const pinchRatio = pinchCount / Math.max(anyHandFrames.length, 1);
  const gapAverage = gapSamples.length ? average(gapSamples) : 0;
  const gapRange = gapSamples.length ? Math.max(...gapSamples) - Math.min(...gapSamples) : 1;
  const stableHold = gapRange <= 0.8;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.12) / 0.74) * 0.18 +
    clamp01((closeHandsRatio - 0.18) / 0.72) * 0.24 +
    clamp01((parallelRatio - 0.12) / 0.76) * 0.12 +
    clamp01((chestLevelRatio - 0.12) / 0.76) * 0.1 +
    clamp01((pinchRatio - 0.14) / 0.72) * 0.2 +
    clamp01((1.08 - gapAverage) / 1.08) * 0.1 +
    clamp01((0.82 - gapRange) / 0.82) * 0.06
  );
  const strictGate =
    (
      twoHandsRatio >= 0.18 &&
      closeHandsRatio >= 0.18 &&
      parallelRatio >= 0.1 &&
      chestLevelRatio >= 0.1 &&
      gapAverage <= 0.92
    ) ||
    pinchRatio >= 0.22;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Маленький" ||
    current.confidence < 0.84 ||
    (
      ["Большой", "Дом", "Дружба", "Любовь"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Маленький" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Маленький" ? current.label : "Большой";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Маленький",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-malenkiy-small-detector",
    scores: [
      { label: "Маленький", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "malenkiy_small_volume_or_pinch",
      detectorScore,
      twoHandsRatio,
      closeHandsRatio,
      chestLevelRatio,
      parallelRatio,
      pinchRatio,
      gapAverage,
      gapRange,
      stableHold,
    },
  };
}

function buildKrasiviyFaceGracePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter((frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.4))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let openPalmCount = 0;
  let nearFaceCount = 0;
  let gracefulEndCount = 0;
  const xSamples = [];
  const ySamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? center;
    const thumbTip = landmarks?.[4] ?? null;
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const face = Array.isArray(frame?.face) ? frame.face : [];
    const headStats = getHeadFrameStats(face, frame?.pose);
    const forehead =
      face?.[10] ?? face?.[151] ?? face?.[9] ?? null;
    const chin = face?.[152] ?? null;
    const leftFace = face?.[234] ?? face?.[93] ?? face?.[132] ?? null;
    const rightFace = face?.[454] ?? face?.[323] ?? face?.[361] ?? null;

    const openPalmLike =
      (
        prediction?.gesture === "Открытая ладонь" &&
        Number(prediction?.confidence ?? 0) >= 0.22
      ) ||
      Number(features?.fingerExtensionAverage ?? 0) >= 0.38 ||
      Number(features?.longFingerRelaxedCount ?? 0) >= 2;

    if (openPalmLike) {
      openPalmCount += 1;
    }

    if (
      !center ||
      !headStats ||
      !forehead ||
      !chin ||
      !leftFace ||
      !rightFace ||
      !noSecondHand
    ) {
      continue;
    }

    const scale = Math.max(Number(headStats.scale ?? 0.1), 0.08);
    const ovalCenter = {
      x: Number(headStats.centerX ?? 0.5),
      y: (Number(forehead.y ?? 0.3) + Number(chin.y ?? 0.6)) / 2,
    };
    const sideOffset =
      Math.abs(Number(center.x ?? 0.5) - Number(ovalCenter.x ?? 0.5)) / scale;
    const verticalNorm =
      (Number(center.y ?? 0.5) - Number(forehead.y ?? 0.3)) /
      Math.max(Number(chin.y ?? 0.6) - Number(forehead.y ?? 0.3), 0.08);
  const nearFace =
      openPalmLike &&
      sideOffset <= 1.18 &&
      verticalNorm >= 0.02 &&
      verticalNorm <= 0.92;

    if (nearFace) {
      nearFaceCount += 1;
      xSamples.push(Number(center.x ?? 0));
      ySamples.push(Number(center.y ?? 0));
      const pinchLike =
        (
          thumbTip &&
          indexTip &&
          getPointDistance(thumbTip, indexTip) / scale <= 0.52
        ) ||
        Number(features?.fingerPinchLike ?? 0) >= 0.18;
      if (pinchLike || Number(features?.fingerExtensionAverage ?? 0) <= 0.54) {
        gracefulEndCount += 1;
      }
    }
  }

  if (ySamples.length < 4) {
    return null;
  }

  const openPalmRatio = openPalmCount / handFrames.length;
  const nearFaceRatio = nearFaceCount / handFrames.length;
  const gracefulEndRatio = gracefulEndCount / Math.max(nearFaceCount, 1);
  const xRangeNorm =
    (Math.max(...xSamples) - Math.min(...xSamples)) / Math.max(average(ySamples.map(() => 0.1)), 0.1);
  const yRangeNorm =
    (Math.max(...ySamples) - Math.min(...ySamples)) / 0.1;
  const downwardSweep = yRangeNorm >= 0.18 && yRangeNorm <= 1.48;
  const softArc = xRangeNorm >= 0.04 && xRangeNorm <= 0.74;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.34) / 0.6) * 0.16 +
    clamp01((openPalmRatio - 0.22) / 0.72) * 0.18 +
    clamp01((nearFaceRatio - 0.2) / 0.74) * 0.28 +
    clamp01((gracefulEndRatio - 0.08) / 0.8) * 0.1 +
    clamp01((yRangeNorm - 0.16) / 1.02) * 0.16 +
    clamp01((0.86 - xRangeNorm) / 0.86) * 0.12
  );
  const strictGate =
    oneHandRatio >= 0.22 &&
    openPalmRatio >= 0.16 &&
    nearFaceRatio >= 0.16 &&
    (downwardSweep || softArc || gracefulEndRatio >= 0.18) &&
    yRangeNorm <= 0.94;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Красивый" ||
    current.confidence < 0.82 ||
    (
      ["Женщина", "Мужчина", "Привет", "Стоп"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Красивый" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Красивый" ? current.label : "Женщина";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Красивый",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-krasiviy-face-grace-detector",
    scores: [
      { label: "Красивый", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "krasiviy_face_grace",
      detectorScore,
      oneHandRatio,
      openPalmRatio,
      nearFaceRatio,
      gracefulEndRatio,
      xRangeNorm,
      yRangeNorm,
      downwardSweep,
      softArc,
      dominantHand,
    },
  };
}

function buildSpasiboThanksPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } = getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.4))) {
    return null;
  }

  let fistCount = 0;
  const fistFrames = [];
  const xSamples = [];
  const ySamples = [];
  const scaleSamples = [];
  const foreheadHitFlags = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!headStats || !noSecondHand) {
      continue;
    }

    const center = getHandCenter(landmarks);
    const face = Array.isArray(frame?.face) ? frame.face : [];
    const forehead = face?.[10] ?? face?.[151] ?? null;

    if (!center || !forehead) {
      continue;
    }

    const scale = Math.max(Number(headStats.scale ?? 0.1), 0.08);
    const gesture = predictGesture(landmarks, handedness);
    const isFist =
      gesture?.gesture === "Кулак" && Number(gesture?.confidence ?? 0) >= 0.52;

    if (!isFist) {
      continue;
    }

    fistCount += 1;
    fistFrames.push(frame);

    const handAnchor = landmarks?.[9] ?? landmarks?.[0] ?? center;
    const foreheadPoint = { x: Number(forehead.x ?? 0.5), y: Number(forehead.y ?? 0.3) + scale * 0.04 };
    const distanceToForehead = getPointDistance(handAnchor, foreheadPoint) / scale;
    const foreheadHit = distanceToForehead <= 1.22;

    foreheadHitFlags.push(foreheadHit);
    xSamples.push(Number(handAnchor.x ?? 0.5));
    ySamples.push(Number(handAnchor.y ?? 0.5));
    scaleSamples.push(scale);
  }

  if (fistFrames.length < 4) {
    return null;
  }

  const scaleAverage = Math.max(average(scaleSamples), 0.08);
  const fistRatio = fistCount / handFrames.length;
  const foreheadIndex = foreheadHitFlags.findIndex(Boolean);
  const foreheadRatio =
    foreheadHitFlags.filter(Boolean).length / Math.max(foreheadHitFlags.length, 1);
  const touchDetected = foreheadIndex >= 0;

  const xRange = (Math.max(...xSamples) - Math.min(...xSamples));
  const xRangeNorm = xRange / scaleAverage;
  const yRange = (Math.max(...ySamples) - Math.min(...ySamples));
  const yRangeNorm = yRange / scaleAverage;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.34) / 0.58) * 0.16 +
    clamp01((fistRatio - 0.22) / 0.78) * 0.22 +
    clamp01((foreheadRatio - 0.14) / 0.76) * 0.28 +
    (touchDetected ? 0.14 : 0) +
    clamp01((2.8 - xRangeNorm) / 2.8) * 0.1 +
    clamp01((2.8 - yRangeNorm) / 2.8) * 0.1,
  );
  const strictGate =
    oneHandRatio >= 0.24 &&
    fistRatio >= 0.18 &&
    touchDetected &&
    foreheadRatio >= 0.14 &&
    xRangeNorm <= 3.2 &&
    yRangeNorm <= 3.2;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Спасибо" ||
    current.confidence < 0.8 ||
    (
      ["Есть", "Пить"].includes(current.label) &&
      current.confidence < 0.84
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Спасибо" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Спасибо" ? current.label : "Есть";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.16),
  );

  return {
    label: "Спасибо",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-spasibo-thanks-detector",
    scores: [
      { label: "Спасибо", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "spasibo_fist_forehead",
      detectorScore,
      oneHandRatio,
      fistRatio,
      foreheadRatio,
      touchDetected,
      xRangeNorm,
      yRangeNorm,
      dominantHand,
    },
  };
}

function buildPrivetSalutePrediction() {
  return null;
}

function buildDruzhbaCrossedIndexPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });

  if (pairedFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let pointingCount = 0;
  let closeTipsCount = 0;
  let crossedCount = 0;
  let poseSupportedFrames = 0;
  let chestLevelFrames = 0;
  const crossedSamples = [];

  for (const frame of pairedFrames) {
    const leftHand = frame?.left_hand;
    const rightHand = frame?.right_hand;
    const leftCenter = getHandCenter(leftHand);
    const rightCenter = getHandCenter(rightHand);

    if (!leftCenter || !rightCenter) {
      continue;
    }

    const leftFeatures = extractHandFeatures(leftHand, "Left");
    const rightFeatures = extractHandFeatures(rightHand, "Right");
    const leftPrediction = predictGesture(leftHand, "Left");
    const rightPrediction = predictGesture(rightHand, "Right");
    const leftIndexPointing =
      (
        leftPrediction?.gesture === "Указательный" &&
        Number(leftPrediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(leftFeatures?.openScores?.index ?? 0) >= 0.62 &&
        Number(leftFeatures?.openScores?.middle ?? 1) <= 0.5 &&
        Number(leftFeatures?.openScores?.ring ?? 1) <= 0.46 &&
        Number(leftFeatures?.openScores?.pinky ?? 1) <= 0.46
      );
    const rightIndexPointing =
      (
        rightPrediction?.gesture === "Указательный" &&
        Number(rightPrediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(rightFeatures?.openScores?.index ?? 0) >= 0.62 &&
        Number(rightFeatures?.openScores?.middle ?? 1) <= 0.5 &&
        Number(rightFeatures?.openScores?.ring ?? 1) <= 0.46 &&
        Number(rightFeatures?.openScores?.pinky ?? 1) <= 0.46
      );
    const bothPointing = leftIndexPointing && rightIndexPointing;

    if (bothPointing) {
      pointingCount += 1;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(shoulderStats?.scale ?? 0.12, 0.08);
    const leftIndexTip = leftHand?.[8] ?? null;
    const rightIndexTip = rightHand?.[8] ?? null;
    const leftIndexPip = leftHand?.[6] ?? null;
    const rightIndexPip = rightHand?.[6] ?? null;

    if (!leftIndexTip || !rightIndexTip || !leftIndexPip || !rightIndexPip) {
      continue;
    }

    const tipsDistance = getPointDistance(leftIndexTip, rightIndexTip) / frameScale;
    const tipsClose = tipsDistance <= 0.62;

    if (tipsClose) {
      closeTipsCount += 1;
    }

    const leftDirection = Number(leftIndexTip.x ?? 0) - Number(leftIndexPip.x ?? 0);
    const rightDirection = Number(rightIndexTip.x ?? 0) - Number(rightIndexPip.x ?? 0);
    const crossingDirection = leftDirection * rightDirection < 0;
    const crossingPips =
      getPointDistance(leftIndexPip, rightIndexPip) / frameScale <= 0.9;
    const crossed = bothPointing && tipsClose && (crossingDirection || crossingPips);

    if (crossed) {
      crossedCount += 1;
      crossedSamples.push({
        x: (Number(leftCenter.x ?? 0) + Number(rightCenter.x ?? 0)) / 2,
        y: (Number(leftCenter.y ?? 0) + Number(rightCenter.y ?? 0)) / 2,
        scale: frameScale,
      });
    }

    if (shoulderStats) {
      poseSupportedFrames += 1;
      const midpointY = (Number(leftCenter.y ?? 0) + Number(rightCenter.y ?? 0)) / 2;
      const minChestY = shoulderStats.centerY - 0.08;
      const maxChestY = shoulderStats.centerY + 0.4;

      if (midpointY >= minChestY && midpointY <= maxChestY) {
        chestLevelFrames += 1;
      }
    }
  }

  if (crossedSamples.length < 4) {
    return null;
  }

  const pointingRatio = pointingCount / pairedFrames.length;
  const closeTipsRatio = closeTipsCount / pairedFrames.length;
  const crossedRatio = crossedCount / pairedFrames.length;
  const chestRatio =
    poseSupportedFrames > 0 ? chestLevelFrames / poseSupportedFrames : 1;
  const scaleAverage = Math.max(
    average(crossedSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = crossedSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = crossedSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const stableHold = xRangeNorm <= 0.22 && yRangeNorm <= 0.26;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.44) / 0.5) * 0.14 +
    clamp01((pointingRatio - 0.28) / 0.66) * 0.2 +
    clamp01((closeTipsRatio - 0.24) / 0.66) * 0.2 +
    clamp01((crossedRatio - 0.2) / 0.7) * 0.24 +
    clamp01((chestRatio - 0.28) / 0.66) * 0.1 +
    clamp01((0.32 - xRangeNorm) / 0.32) * 0.06 +
    clamp01((0.34 - yRangeNorm) / 0.34) * 0.04 +
    (stableHold ? 0.02 : 0),
  );
  const strictGate =
    twoHandsRatio >= 0.56 &&
    pointingRatio >= 0.48 &&
    closeTipsRatio >= 0.4 &&
    crossedRatio >= 0.34 &&
    (poseSupportedFrames < 3 || chestRatio >= 0.38);

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Дружба" ||
    current.confidence < 0.84 ||
    (
      ["Мужчина", "Женщина", "Привет", "Пока", "Дом", "Солнце", "Да", "Нет"].includes(
        current.label,
      ) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Дружба" ? current.confidence : 0,
    0.69 + detectorScore * 0.25,
  ));
  const secondaryLabel =
    current.label && current.label !== "Дружба" ? current.label : "Дом";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Дружба",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-druzhba-cross-index-detector",
    scores: [
      { label: "Дружба", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "druzhba_crossed_index_fingers",
      detectorScore,
      twoHandsRatio,
      pointingRatio,
      closeTipsRatio,
      crossedRatio,
      chestRatio,
      xRangeNorm,
      yRangeNorm,
      stableHold,
    },
  };
}

function buildZhenshchinaCheekPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let openPalmCount = 0;
  let nearCheekCount = 0;
  let lowerFaceCount = 0;
  const xSamples = [];
  const ySamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? center;
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);
    const openOrPoint =
      (
        prediction?.gesture === "Указательный" ||
        prediction?.gesture === "Открытая ладонь"
      ) &&
      Number(prediction?.confidence ?? 0) >= 0.3;

    const openPalmLike =
      openOrPoint ||
      Number(features?.fingerExtensionAverage ?? 0) >= 0.52 ||
      Number(features?.longFingerRelaxedCount ?? 0) >= 2;

    if (openPalmLike) {
      openPalmCount += 1;
    }

    const face = Array.isArray(frame?.face) ? frame.face : [];
    const nose = getFaceFeaturePoint(face, [1, 4, 5]);
    const leftCheek = face?.[78] ?? null;
    const rightCheek = face?.[308] ?? null;
    const chin = face?.[152] ?? null;

    if (!indexTip || !nose || !chin || !noSecondHand) {
      continue;
    }

    const cheekPoint = dominantHand === "left" ? leftCheek : rightCheek;

    if (!cheekPoint) {
      continue;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const scale = Math.max(shoulderStats?.scale ?? getPointDistance(nose, chin) ?? 0.08, 0.08);
    const targetZone = {
      x: Number(cheekPoint.x ?? 0) * 0.62 + Number(chin.x ?? 0) * 0.38,
      y: Number(cheekPoint.y ?? 0) * 0.58 + Number(chin.y ?? 0) * 0.42,
    };
    const handAnchor = center ?? indexTip;
    const distanceToCheek = getPointDistance(handAnchor, targetZone) / scale;
    const lowerFace = Number(handAnchor.y ?? 0) >= Number(nose.y ?? 0) - 0.02;
    const nearCheek = openPalmLike && distanceToCheek <= 0.9 && lowerFace;

    if (nearCheek) {
      nearCheekCount += 1;
    }

    if (lowerFace) {
      lowerFaceCount += 1;
    }

    xSamples.push(Number(handAnchor.x ?? 0) / scale);
    ySamples.push(Number(handAnchor.y ?? 0) / scale);
  }

  if (ySamples.length < 4 || xSamples.length < 4) {
    return null;
  }

  const openPalmRatio = openPalmCount / handFrames.length;
  const nearCheekRatio = nearCheekCount / handFrames.length;
  const lowerFaceRatio = lowerFaceCount / handFrames.length;
  const xRangeNorm =
    (Math.max(...xSamples) - Math.min(...xSamples)) / Math.max(
      average(xSamples.map((sample) => Math.abs(sample))) || 0.08,
      0.08,
    );
  const yRangeNorm =
    (Math.max(...ySamples) - Math.min(...ySamples)) / Math.max(
      average(ySamples.map((sample) => Math.abs(sample))) || 0.08,
      0.08,
    );
  const xDirectionChanges = countDirectionChanges(xSamples, 0.0018);
  const downwardSweep =
    yRangeNorm >= 0.16 &&
    yRangeNorm <= 0.92 &&
    xRangeNorm <= 0.44;
  const sideSweep =
    xRangeNorm >= 0.08 &&
    xRangeNorm <= 0.64 &&
    yRangeNorm <= 0.98 &&
    (xDirectionChanges >= 1 || downwardSweep);
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.48) / 0.4) * 0.16 +
    clamp01((openPalmRatio - 0.34) / 0.54) * 0.18 +
    clamp01((nearCheekRatio - 0.3) / 0.5) * 0.24 +
    clamp01((lowerFaceRatio - 0.42) / 0.46) * 0.16 +
    clamp01((0.46 - xRangeNorm) / 0.46) * 0.08 +
    clamp01((yRangeNorm - 0.12) / 0.52) * 0.08 +
    clamp01(xDirectionChanges / 3) * 0.08,
  );
  const strictGate =
    oneHandRatio >= 0.4 &&
    openPalmRatio >= 0.22 &&
    nearCheekRatio >= 0.22 &&
    lowerFaceRatio >= 0.3 &&
    (sideSweep || downwardSweep || nearCheekRatio >= 0.34);

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Женщина" ||
    current.confidence < 0.82 ||
    (
      current.label === "Мужчина" &&
      current.confidence < 0.82
    ) ||
    (
      ["Привет", "Пока", "Дом", "Солнце", "Да", "Нет"].includes(current.label) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Женщина" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Женщина" ? current.label : "Мужчина";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.75 - detectorScore * 0.16),
  );

  return {
    label: "Женщина",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-zhenshchina-cheek-detector",
    scores: [
      { label: "Женщина", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "zhenshchina_cheek_brush",
      detectorScore,
      oneHandRatio,
      openPalmRatio,
      nearCheekRatio,
      lowerFaceRatio,
      xRangeNorm,
      yRangeNorm,
      xDirectionChanges,
      downwardSweep,
      sideSweep,
      dominantHand,
    },
  };
}

function buildZhenshchinaPalmSwipePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } =
    getDominantHandFrameSet(frames);
  const directionSign = dominantHand === "left" ? -1 : 1;

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  let openPalmCount = 0;
  let fingersTogetherCount = 0;
  let nearCheekCount = 0;
  let leftCheekTouchCount = 0;
  let rightCheekTouchCount = 0;
  let cheekSwitchTransitions = 0;
  let lowerFaceCount = 0;
  let sidePalmCount = 0;
  const xSamples = [];
  const ySamples = [];
  const cheekSideStates = [];
  const sidePalmSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const features = extractHandFeatures(landmarks, handedness);
    const prediction = predictGesture(landmarks, handedness);
    const center = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? center;
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!features || !indexTip || !noSecondHand) {
      continue;
    }

    const openPalmDetected =
      (
        prediction?.gesture === "Открытая ладонь" &&
        Number(prediction?.confidence ?? 0) >= 0.34
      ) ||
      (
        Number(features.longFingerRelaxedCount ?? 0) >= 3 &&
        Number(features.fingerExtensionAverage ?? 0) >= 0.5
      );

    if (openPalmDetected) {
      openPalmCount += 1;
    }

    const averageFingerGap =
      (Number(features.indexMiddleGap ?? 1) +
        Number(features.middleRingGap ?? 1) +
        Number(features.ringPinkyGap ?? 1)) /
      3;
    const fingersTogether = openPalmDetected && averageFingerGap <= 0.6;

    if (fingersTogether) {
      fingersTogetherCount += 1;
    }

    const face = Array.isArray(frame?.face) ? frame.face : [];
    const headStats = getHeadFrameStats(face, frame?.pose);
    const nose = getFaceFeaturePoint(face, [1, 4, 5]);
    const leftCheek = face?.[78] ?? null;
    const rightCheek = face?.[308] ?? null;
    const chin = face?.[152] ?? null;
    const cheekPoint = dominantHand === "left" ? leftCheek : rightCheek;
    const hasFacePoints = Boolean(nose && chin && cheekPoint);

    if (!headStats && !hasFacePoints) {
      continue;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const scale = Math.max(
      shoulderStats?.scale ??
        (hasFacePoints ? getPointDistance(nose, chin) : 0) ??
        Number(headStats?.scale ?? 0.1),
      0.08,
    );
    const faceCenterX = Number(
      (hasFacePoints ? nose.x : headStats?.centerX) ?? 0.5,
    );
    const faceEyeY = Number(
      (hasFacePoints ? nose.y : headStats?.eyeY) ?? 0.34,
    );
    const approxNose = {
      x: faceCenterX,
      y: hasFacePoints ? Number(nose.y ?? faceEyeY) : faceEyeY + scale * 0.18,
    };
    const approxChin = {
      x: hasFacePoints ? Number(chin.x ?? faceCenterX) : faceCenterX,
      y: hasFacePoints ? Number(chin.y ?? faceEyeY + scale * 1.18) : faceEyeY + scale * 1.18,
    };
    const approxCheek = hasFacePoints
      ? { x: Number(cheekPoint.x ?? faceCenterX), y: Number(cheekPoint.y ?? faceEyeY + scale * 0.55) }
      : { x: faceCenterX + directionSign * scale * 0.55, y: faceEyeY + scale * 0.55 };
    const approxLeftCheek = hasFacePoints
      ? { x: Number(leftCheek?.x ?? faceCenterX - scale * 0.55), y: Number(leftCheek?.y ?? faceEyeY + scale * 0.55) }
      : { x: faceCenterX - scale * 0.55, y: faceEyeY + scale * 0.55 };
    const approxRightCheek = hasFacePoints
      ? { x: Number(rightCheek?.x ?? faceCenterX + scale * 0.55), y: Number(rightCheek?.y ?? faceEyeY + scale * 0.55) }
      : { x: faceCenterX + scale * 0.55, y: faceEyeY + scale * 0.55 };
    const targetZone = {
      x: approxCheek.x * 0.6 + approxChin.x * 0.4,
      y: approxCheek.y * 0.46 + approxChin.y * 0.54,
    };
    const handAnchor = center ?? indexTip;
    const distanceToCheek = getPointDistance(handAnchor, targetZone) / scale;
    const lowerFace =
      Number(handAnchor.y ?? 0) >= Number(approxNose.y ?? faceEyeY) - scale * 0.18;
    const nearCheek = openPalmDetected && distanceToCheek <= 1.42 && lowerFace;

    if (nearCheek) {
      nearCheekCount += 1;
      const distanceToLeft = getPointDistance(handAnchor, approxLeftCheek) / scale;
      const distanceToRight = getPointDistance(handAnchor, approxRightCheek) / scale;
      const cheekSide = distanceToLeft <= distanceToRight ? "left" : "right";

      cheekSideStates.push(cheekSide);
      if (cheekSide === "left") {
        leftCheekTouchCount += 1;
      } else {
        rightCheekTouchCount += 1;
      }

      const palmSize = Number(features?.palmSize ?? 0);
      const palmWidth = Number(features?.palmWidth ?? 0);
      const palmWidthRatio = palmSize > 0 ? palmWidth / palmSize : 1;
      const sidePalm = palmWidthRatio <= 0.78 || Number(features?.palmSpread ?? 1) <= 0.42;

      sidePalmSamples.push(palmWidthRatio);
      if (sidePalm) {
        sidePalmCount += 1;
      }
    }

    if (lowerFace) {
      lowerFaceCount += 1;
    }

    xSamples.push((Number(handAnchor.x ?? 0) - faceCenterX) / scale);
    ySamples.push((Number(handAnchor.y ?? 0) - faceEyeY) / scale);
  }

  if (ySamples.length < 4 || xSamples.length < 4) {
    return null;
  }

  const openPalmRatio = openPalmCount / handFrames.length;
  const fingersTogetherRatio = fingersTogetherCount / handFrames.length;
  const nearCheekRatio = nearCheekCount / handFrames.length;
  const lowerFaceRatio = lowerFaceCount / handFrames.length;
  const sidePalmRatio = nearCheekCount > 0 ? sidePalmCount / nearCheekCount : 0;
  const palmWidthRatioAverage = sidePalmSamples.length ? average(sidePalmSamples) : 1;
  const xRangeNorm = Math.max(...xSamples) - Math.min(...xSamples);
  const yRangeNorm = Math.max(...ySamples) - Math.min(...ySamples);
  const chunkSize = Math.max(2, Math.round(cheekSideStates.length * 0.34));
  const earlySide = cheekSideStates.length
    ? cheekSideStates.slice(0, chunkSize).filter(Boolean)[0] ?? ""
    : "";
  const lateSide = cheekSideStates.length
    ? cheekSideStates.slice(-chunkSize).filter(Boolean)[0] ?? ""
    : "";
  let previousCheekSide = cheekSideStates[0] ?? "";
  for (let index = 1; index < cheekSideStates.length; index += 1) {
    const nextSide = cheekSideStates[index];

    if (nextSide && previousCheekSide && nextSide !== previousCheekSide) {
      cheekSwitchTransitions += 1;
      previousCheekSide = nextSide;
    } else if (nextSide) {
      previousCheekSide = nextSide;
    }
  }

  const cheekSwitchGesture =
    leftCheekTouchCount >= 1 &&
    rightCheekTouchCount >= 1 &&
    (
      cheekSwitchTransitions >= 1 ||
      (earlySide && lateSide && earlySide !== lateSide)
    ) &&
    xRangeNorm >= 0.28;

  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.42) / 0.5) * 0.12 +
      clamp01((openPalmRatio - 0.22) / 0.7) * 0.18 +
      clamp01((fingersTogetherRatio - 0.14) / 0.7) * 0.12 +
      clamp01((nearCheekRatio - 0.18) / 0.7) * 0.28 +
      clamp01((lowerFaceRatio - 0.24) / 0.76) * 0.12 +
      clamp01((xRangeNorm - 0.5) / 1.2) * 0.1 +
      clamp01(cheekSwitchTransitions / 2) * 0.08 +
      clamp01((sidePalmRatio - 0.24) / 0.76) * 0.1,
  );

  const strictGate =
    oneHandRatio >= 0.32 &&
    openPalmRatio >= 0.18 &&
    nearCheekRatio >= 0.12 &&
    lowerFaceRatio >= 0.18 &&
    cheekSwitchGesture &&
    sidePalmRatio >= 0.22;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Женщина" ||
    current.confidence < 0.84 ||
    (["Мужчина", "Есть", "Спасибо", "Пить", "Пока", "Привет"].includes(current.label) &&
      current.confidence < 0.92);

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(
    Math.max(current.label === "Женщина" ? current.confidence : 0, 0.72 + detectorScore * 0.24),
  );
  const secondaryLabel = current.label && current.label !== "Женщина" ? current.label : "Мужчина";
  const secondaryConfidence = clamp01(Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16));

  return {
    label: "Женщина",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-zhenshchina-palm-swipe-detector",
    scores: [
      { label: "Женщина", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "zhenshchina_cheek_switch_palm_side",
      detectorScore,
      oneHandRatio,
      openPalmRatio,
      fingersTogetherRatio,
      nearCheekRatio,
      lowerFaceRatio,
      sidePalmRatio,
      palmWidthRatioAverage,
      xRangeNorm,
      yRangeNorm,
      leftCheekTouchCount,
      rightCheekTouchCount,
      cheekSwitchTransitions,
      dominantHand,
    },
  };
}

function buildDruzhbaInterlockPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });

  if (pairedFrames.length < Math.max(6, Math.round(frames.length * 0.42))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let bothFistCount = 0;
  let closeCount = 0;
  let interlockCount = 0;
  let poseSupportedFrames = 0;
  let chestLevelFrames = 0;
  const interlockSamples = [];

  for (const frame of pairedFrames) {
    const leftHand = frame?.left_hand;
    const rightHand = frame?.right_hand;
    const leftCenter = getHandCenter(leftHand);
    const rightCenter = getHandCenter(rightHand);

    if (!leftCenter || !rightCenter) {
      continue;
    }

    const leftFeatures = extractHandFeatures(leftHand, "Left");
    const rightFeatures = extractHandFeatures(rightHand, "Right");
    const leftPrediction = predictGesture(leftHand, "Left");
    const rightPrediction = predictGesture(rightHand, "Right");
    const leftFist =
      (leftPrediction?.gesture === "Кулак" &&
        Number(leftPrediction?.confidence ?? 0) >= 0.24) ||
      (
        Number(leftFeatures?.curledLongFingerCount ?? 0) >= 2 &&
        Number(leftFeatures?.fingerExtensionAverage ?? 1) <= 0.66
      );
    const rightFist =
      (rightPrediction?.gesture === "Кулак" &&
        Number(rightPrediction?.confidence ?? 0) >= 0.24) ||
      (
        Number(rightFeatures?.curledLongFingerCount ?? 0) >= 2 &&
        Number(rightFeatures?.fingerExtensionAverage ?? 1) <= 0.66
      );
    const bothFists = leftFist && rightFist;

    if (bothFists) {
      bothFistCount += 1;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = shoulderStats?.scale ?? 0.12;
    const centerDistance =
      getPointDistance(leftCenter, rightCenter) / Math.max(frameScale, 0.08);
    const midpoint = {
      x: (Number(leftCenter.x ?? 0) + Number(rightCenter.x ?? 0)) / 2,
      y: (Number(leftCenter.y ?? 0) + Number(rightCenter.y ?? 0)) / 2,
    };
    const closeHands = centerDistance <= 1.52;

    if (closeHands) {
      closeCount += 1;
    }

    if (shoulderStats) {
      poseSupportedFrames += 1;
      const minChestY = shoulderStats.centerY - 0.08;
      const maxChestY = shoulderStats.centerY + 0.36;

      if (midpoint.y >= minChestY && midpoint.y <= maxChestY) {
        chestLevelFrames += 1;
      }
    }

    const fingerMeshLike =
      centerDistance <= 1.64 &&
      Math.abs(Number(leftCenter.y ?? 0.5) - Number(rightCenter.y ?? 0.5)) / Math.max(frameScale, 0.08) <= 0.86;
    const interlocked = (bothFists || fingerMeshLike) && closeHands;

    if (interlocked) {
      interlockCount += 1;
      interlockSamples.push({
        x: midpoint.x,
        y: midpoint.y,
      });
    }
  }

  if (interlockSamples.length < 4) {
    return null;
  }

  const bothFistRatio = bothFistCount / pairedFrames.length;
  const closeRatio = closeCount / pairedFrames.length;
  const interlockRatio = interlockCount / pairedFrames.length;
  const chestRatio =
    poseSupportedFrames > 0 ? chestLevelFrames / poseSupportedFrames : 1;
  const xValues = interlockSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = interlockSamples.map((sample) => Number(sample.y ?? 0));
  const xRange = xValues.length >= 2 ? Math.max(...xValues) - Math.min(...xValues) : 0;
  const yRange = yValues.length >= 2 ? Math.max(...yValues) - Math.min(...yValues) : 0;
  const directionChanges = countDirectionChanges(xValues, 0.0016);
  const fixedHold = xRange <= 0.1 && yRange <= 0.1 && interlockRatio >= 0.36;
  const gentleSway =
    xRange >= 0.01 && xRange <= 0.24 && yRange <= 0.2 && directionChanges <= 3;
  const stableBondGesture = fixedHold || gentleSway;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.3) / 0.62) * 0.15 +
    clamp01((bothFistRatio - 0.16) / 0.78) * 0.16 +
    clamp01((closeRatio - 0.18) / 0.76) * 0.2 +
    clamp01((interlockRatio - 0.14) / 0.78) * 0.24 +
    clamp01((chestRatio - 0.18) / 0.78) * 0.1 +
    clamp01((0.42 - xRange) / 0.42) * 0.08 +
    clamp01((0.28 - yRange) / 0.28) * 0.05 +
    (stableBondGesture ? 0.04 : 0.01),
  );
  const strictGate =
    twoHandsRatio >= 0.32 &&
    closeRatio >= 0.18 &&
    interlockRatio >= 0.16 &&
    (poseSupportedFrames < 3 || chestRatio >= 0.2) &&
    stableBondGesture;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Дружба" ||
    current.confidence < 0.8 ||
    (
      ["Мужчина", "Женщина", "Привет", "Пока", "Дом", "Солнце", "Да", "Нет"].includes(
        current.label,
      ) &&
      current.confidence < 0.9
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Дружба" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Дружба" ? current.label : "Привет";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Дружба",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-druzhba-interlock-detector",
    scores: [
      { label: "Дружба", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "druzhba_interlocked_fists",
      detectorScore,
      twoHandsRatio,
      bothFistRatio,
      closeRatio,
      interlockRatio,
      chestRatio,
      xRange,
      yRange,
      directionChanges,
      fixedHold,
      gentleSway,
    },
  };
}

function buildSolntseSunrayPrediction({
  frames = [],
  metadata = null,
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand =
    availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks:
        dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite:
        dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.45))) {
    return null;
  }

  let oneHandCount = 0;
  let openPalmCount = 0;
  let aboveHeadCount = 0;
  let sunPoseCount = 0;
  const motionSamples = [];

  for (const frame of handFrames) {
    const oppositeCount =
      Array.isArray(frame.opposite) && frame.opposite.length >= 21 ? 1 : 0;
    if (oppositeCount === 0) {
      oneHandCount += 1;
    }

    const landmarks = frame.landmarks;
    const center = getHandCenter(landmarks);

    if (!center) {
      continue;
    }

    const gesturePrediction = predictGesture(landmarks, handedness);
    const features = extractHandFeatures(landmarks, handedness);
    const openPalm =
      (
        gesturePrediction?.gesture === "Открытая ладонь" &&
        Number(gesturePrediction?.confidence ?? 0) >= 0.42
      ) ||
      (
        Number(features?.longFingerRelaxedCount ?? 0) >= 3 &&
        Number(features?.fingerExtensionAverage ?? 0) >= 0.62
      );

    if (openPalm) {
      openPalmCount += 1;
    }

    const headStats = getHeadFrameStats(frame.face, frame.pose);

    if (!headStats) {
      continue;
    }

    const normalizedDistanceToHeadCenter =
      Math.abs(Number(center.x ?? 0) - Number(headStats.centerX ?? 0)) /
      Math.max(Number(headStats.scale ?? 0.1), 0.08);
    const aboveHead =
      Number(center.y ?? 1) <= Number(headStats.eyeY ?? 0.35) + 0.11 &&
      Number(center.y ?? 1) <= Number(headStats.topY ?? 0.3) + 0.2;
    const sideReach = normalizedDistanceToHeadCenter <= 1.9;

    if (aboveHead) {
      aboveHeadCount += 1;
    }

    if (openPalm && aboveHead && sideReach) {
      sunPoseCount += 1;
      motionSamples.push({
        x: Number(center.x ?? 0),
        y: Number(center.y ?? 0),
        scale: Number(headStats.scale ?? 0.1),
      });
    }
  }

  if (motionSamples.length < 4) {
    return null;
  }

  const oneHandRatio = oneHandCount / handFrames.length;
  const openPalmRatio = openPalmCount / handFrames.length;
  const aboveHeadRatio = aboveHeadCount / handFrames.length;
  const sunPoseRatio = sunPoseCount / handFrames.length;
  const scaleAverage = Math.max(
    average(motionSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = motionSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = motionSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const xDirectionChanges = countDirectionChanges(xValues, 0.0014);
  const yDirectionChanges = countDirectionChanges(yValues, 0.0014);
  const circularMotion =
    xRangeNorm >= 0.08 &&
    yRangeNorm >= 0.06 &&
    xDirectionChanges >= 1 &&
    yDirectionChanges >= 1;
  const glowSwayMotion =
    (xRangeNorm >= 0.08 || yRangeNorm >= 0.06) &&
    xRangeNorm <= 0.62 &&
    yRangeNorm <= 0.62 &&
    xDirectionChanges + yDirectionChanges >= 1;
  const motionGate = circularMotion || glowSwayMotion;
  const waveScore = Number(metadata?.waveScore ?? 0);
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.5) / 0.4) * 0.18 +
    clamp01((openPalmRatio - 0.36) / 0.52) * 0.2 +
    clamp01((aboveHeadRatio - 0.34) / 0.56) * 0.22 +
    clamp01((sunPoseRatio - 0.28) / 0.62) * 0.22 +
    clamp01((xRangeNorm - 0.06) / 0.42) * 0.08 +
    clamp01((yRangeNorm - 0.05) / 0.38) * 0.06 +
    (circularMotion ? 0.03 : 0) +
    (glowSwayMotion ? 0.01 : 0),
  );
  const strictGate =
    oneHandRatio >= 0.56 &&
    openPalmRatio >= 0.44 &&
    aboveHeadRatio >= 0.38 &&
    sunPoseRatio >= 0.34 &&
    motionGate;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Солнце" ||
    current.confidence < 0.8 ||
    (
      ["Привет", "Пока", "Женщина", "Дом", "Да", "Нет"].includes(current.label) &&
      (
        current.confidence < 0.9 ||
        (aboveHeadRatio >= 0.58 && waveScore < 0.72)
      )
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Солнце" ? current.confidence : 0,
    0.68 + detectorScore * 0.27,
  ));
  const secondaryLabel =
    current.label && current.label !== "Солнце" ? current.label : "Привет";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Солнце",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-solntse-sunray-detector",
    scores: [
      { label: "Солнце", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "solntse_open_palm_above_head",
      detectorScore,
      oneHandRatio,
      openPalmRatio,
      aboveHeadRatio,
      sunPoseRatio,
      xRangeNorm,
      yRangeNorm,
      xDirectionChanges,
      yDirectionChanges,
      circularMotion,
      glowSwayMotion,
      waveScore,
      dominantHand,
    },
  };
}

function buildDomRoofPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const pairedFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft && hasRight;
  });

  if (pairedFrames.length < Math.max(6, Math.round(frames.length * 0.4))) {
    return null;
  }

  const twoHandsRatio = pairedFrames.length / frames.length;
  let openHandsCount = 0;
  let apexCloseCount = 0;
  let roofShapeCount = 0;
  let poseSupportedFrames = 0;
  let chestLevelFrames = 0;
  const roofSamples = [];

  for (const frame of pairedFrames) {
    const leftHand = frame?.left_hand;
    const rightHand = frame?.right_hand;
    const leftCenter = getHandCenter(leftHand);
    const rightCenter = getHandCenter(rightHand);

    if (!leftCenter || !rightCenter) {
      continue;
    }

    const leftFeatures = extractHandFeatures(leftHand, "Left");
    const rightFeatures = extractHandFeatures(rightHand, "Right");
    const leftOpen =
      Number(leftFeatures?.fingerExtensionAverage ?? 0) >= 0.46 ||
      Number(leftFeatures?.longFingerRelaxedCount ?? 0) >= 2;
    const rightOpen =
      Number(rightFeatures?.fingerExtensionAverage ?? 0) >= 0.46 ||
      Number(rightFeatures?.longFingerRelaxedCount ?? 0) >= 2;
    const bothOpenHands = leftOpen && rightOpen;

    if (bothOpenHands) {
      openHandsCount += 1;
    }

    const shoulderStats = getShoulderFrameStats(frame?.pose);
    const frameScale = Math.max(shoulderStats?.scale ?? 0.12, 0.08);
    const leftIndexTip = leftHand?.[8] ?? null;
    const rightIndexTip = rightHand?.[8] ?? null;
    const leftMiddleTip = leftHand?.[12] ?? null;
    const rightMiddleTip = rightHand?.[12] ?? null;
    const leftThumbTip = leftHand?.[4] ?? null;
    const rightThumbTip = rightHand?.[4] ?? null;
    const leftWrist = leftHand?.[0] ?? null;
    const rightWrist = rightHand?.[0] ?? null;

    if (
      !leftIndexTip ||
      !rightIndexTip ||
      !leftThumbTip ||
      !rightThumbTip ||
      !leftWrist ||
      !rightWrist
    ) {
      continue;
    }

    const apexDistance = getPointDistance(leftIndexTip, rightIndexTip) / frameScale;
    const thumbDistance = getPointDistance(leftThumbTip, rightThumbTip) / frameScale;
    const middleDistance =
      getPointDistance(leftMiddleTip ?? leftIndexTip, rightMiddleTip ?? rightIndexTip) /
      frameScale;
    const handsCenterDistance = getPointDistance(leftCenter, rightCenter) / frameScale;
    const apexClose = apexDistance <= 0.64;
    const roofSpan = thumbDistance >= Math.max(0.34, apexDistance * 1.14);
    const roofRidge = middleDistance <= Math.max(1.18 * thumbDistance, 0.95);
    const roofTopAboveWrists =
      Number(leftIndexTip.y ?? 1) <= Number(leftWrist.y ?? 0) + 0.02 &&
      Number(rightIndexTip.y ?? 1) <= Number(rightWrist.y ?? 0) + 0.02;
    const roofShape =
      apexClose &&
      roofSpan &&
      roofRidge &&
      roofTopAboveWrists &&
      handsCenterDistance <= 1.42 &&
      (bothOpenHands || middleDistance <= 0.78);

    if (apexClose) {
      apexCloseCount += 1;
    }

    if (roofShape) {
      roofShapeCount += 1;
      roofSamples.push({
        x: (Number(leftCenter.x ?? 0) + Number(rightCenter.x ?? 0)) / 2,
        y: (Number(leftCenter.y ?? 0) + Number(rightCenter.y ?? 0)) / 2,
        scale: frameScale,
      });
    }

    if (shoulderStats) {
      poseSupportedFrames += 1;
      const midpointY = (Number(leftCenter.y ?? 0) + Number(rightCenter.y ?? 0)) / 2;
      const minChestY = shoulderStats.centerY - 0.1;
      const maxChestY = shoulderStats.centerY + 0.44;

      if (midpointY >= minChestY && midpointY <= maxChestY) {
        chestLevelFrames += 1;
      }
    }
  }

  if (roofSamples.length < 4) {
    return null;
  }

  const openHandsRatio = openHandsCount / pairedFrames.length;
  const apexCloseRatio = apexCloseCount / pairedFrames.length;
  const roofShapeRatio = roofShapeCount / pairedFrames.length;
  const chestRatio =
    poseSupportedFrames > 0 ? chestLevelFrames / poseSupportedFrames : 1;
  const scaleAverage = Math.max(
    average(roofSamples.map((sample) => Number(sample.scale ?? 0.1))),
    0.08,
  );
  const xValues = roofSamples.map((sample) => Number(sample.x ?? 0));
  const yValues = roofSamples.map((sample) => Number(sample.y ?? 0));
  const xRangeNorm =
    (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm =
    (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const xDirectionChanges = countDirectionChanges(xValues, 0.0016);
  const yDirectionChanges = countDirectionChanges(yValues, 0.0016);
  const stableRoofHold = xRangeNorm <= 0.18 && yRangeNorm <= 0.22;
  const wallMotionHint =
    yRangeNorm >= 0.05 &&
    yRangeNorm <= 0.32 &&
    yDirectionChanges <= 2 &&
    xRangeNorm <= 0.22;
  const roofMotionGate = stableRoofHold || wallMotionHint;
  const detectorScore = clamp01(
    clamp01((twoHandsRatio - 0.44) / 0.5) * 0.16 +
    clamp01((openHandsRatio - 0.3) / 0.64) * 0.1 +
    clamp01((apexCloseRatio - 0.3) / 0.62) * 0.2 +
    clamp01((roofShapeRatio - 0.24) / 0.66) * 0.24 +
    clamp01((chestRatio - 0.28) / 0.66) * 0.12 +
    clamp01((0.32 - xRangeNorm) / 0.32) * 0.06 +
    clamp01((0.34 - yRangeNorm) / 0.34) * 0.07 +
    (stableRoofHold ? 0.03 : 0) +
    (wallMotionHint ? 0.02 : 0),
  );
  const strictGate =
    twoHandsRatio >= 0.56 &&
    openHandsRatio >= 0.38 &&
    apexCloseRatio >= 0.36 &&
    roofShapeRatio >= 0.32 &&
    (poseSupportedFrames < 3 || chestRatio >= 0.4) &&
    roofMotionGate;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Дом" ||
    current.confidence < 0.82 ||
    (
      ["Дружба", "Женщина", "Мужчина", "Привет", "Пока", "Солнце", "Да", "Нет"].includes(
        current.label,
      ) &&
      current.confidence < 0.92
    );

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Дом" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Дом" ? current.label : "Дружба";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.76 - detectorScore * 0.16),
  );

  return {
    label: "Дом",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-dom-roof-detector",
    scores: [
      { label: "Дом", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "dom_roof_triangle",
      detectorScore,
      twoHandsRatio,
      openHandsRatio,
      apexCloseRatio,
      roofShapeRatio,
      chestRatio,
      xRangeNorm,
      yRangeNorm,
      xDirectionChanges,
      yDirectionChanges,
      stableRoofHold,
      wallMotionHint,
    },
  };
}

function buildMuzhchinaMoustachePrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const availableLeftFrames = frames.filter(
    (frame) => Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21,
  );
  const availableRightFrames = frames.filter(
    (frame) => Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21,
  );
  const dominantHand = availableRightFrames.length >= availableLeftFrames.length ? "right" : "left";
  const handedness = dominantHand === "left" ? "Left" : "Right";
  const handFrames = frames
    .map((frame) => ({
      landmarks: dominantHand === "left" ? frame?.left_hand : frame?.right_hand,
      opposite: dominantHand === "left" ? frame?.right_hand : frame?.left_hand,
      face: frame?.face,
      pose: frame?.pose,
    }))
    .filter(
      (frame) => Array.isArray(frame.landmarks) && frame.landmarks.length >= 21,
    );

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.4))) {
    return null;
  }

  const oneHandFrames = frames.filter((frame) => {
    const hasLeft = Array.isArray(frame?.left_hand) && frame.left_hand.length >= 21;
    const hasRight = Array.isArray(frame?.right_hand) && frame.right_hand.length >= 21;
    return hasLeft !== hasRight;
  }).length;
  const oneHandRatio = oneHandFrames / frames.length;
  let nearForeheadCount = 0;
  let upperFaceCount = 0;
  let touchLikeCount = 0;
  const ySamples = [];
  const outwardSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const center = getHandCenter(landmarks);
    const indexTip = landmarks?.[8] ?? center;
    const thumbTip = landmarks?.[4] ?? null;
    const headStats = getHeadFrameStats(frame?.face, frame?.pose);
    const noSecondHand =
      !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!indexTip || !headStats || !noSecondHand) {
      continue;
    }

    const face = Array.isArray(frame?.face) ? frame.face : [];
    const forehead = face?.[10] ?? null;
    const nose = getFaceFeaturePoint(face, [1, 4, 5]);

    if (!forehead || !nose) {
      continue;
    }

    const features = extractHandFeatures(landmarks, handedness);
    const openOrPoint =
      Number(features?.openScores?.index ?? 0) >= 0.28 ||
      Number(features?.longFingerRelaxedCount ?? 0) >= 2 ||
      Number(features?.fingerExtensionAverage ?? 0) >= 0.34;
    const scale = Math.max(Number(headStats.scale ?? 0.08), 0.08);
    const handAnchor = center && indexTip
      ? {
        x: Number(indexTip.x ?? 0) * 0.68 + Number(center.x ?? 0) * 0.32,
        y: Number(indexTip.y ?? 0) * 0.68 + Number(center.y ?? 0) * 0.32,
      }
      : (center ?? indexTip);
    const foreheadPoint = {
      x: Number(forehead.x ?? 0.5),
      y: Number(forehead.y ?? 0.3) + scale * 0.04,
    };
    const distanceToForehead = getPointDistance(handAnchor, foreheadPoint) / scale;
    const upperFace =
      Number(handAnchor.y ?? 0.5) <= Number(nose.y ?? 0.45) + scale * 0.2;
    const touchLike =
      (thumbTip && getPointDistance(thumbTip, indexTip) / scale <= 0.7) ||
      Number(features?.fingerPinchLike ?? 0) >= 0.12 ||
      openOrPoint;
    const nearForehead =
      touchLike &&
      distanceToForehead <= 1.08 &&
      Math.abs(Number(handAnchor.x ?? 0.5) - Number(foreheadPoint.x ?? 0.5)) / scale <= 0.96;

    if (nearForehead) {
      nearForeheadCount += 1;
      ySamples.push(Number(handAnchor.y ?? 0.5) / scale);
    }

    if (upperFace) {
      upperFaceCount += 1;
    }

    if (touchLike) {
      touchLikeCount += 1;
    }

    const outward =
      (Number(handAnchor.y ?? 0.5) - Number(foreheadPoint.y ?? 0.3)) / scale;
    outwardSamples.push(outward);
  }

  if (outwardSamples.length < 4) {
    return null;
  }

  const nearRatio = nearForeheadCount / handFrames.length;
  const gripLikeRatio = upperFaceCount / handFrames.length;
  const pinchRatio = touchLikeCount / handFrames.length;
  const proximityScore = clamp01(1 - Math.max(0, 0.92 - nearRatio));
  const lipProximityScore = gripLikeRatio;
  const chunkSize = Math.max(2, Math.round(outwardSamples.length * 0.35));
  const startY = average(outwardSamples.slice(0, chunkSize));
  const endY = average(outwardSamples.slice(-chunkSize));
  const downwardDelta = endY - startY;
  const yRange = ySamples.length >= 2
    ? Math.max(...ySamples) - Math.min(...ySamples)
    : 0;
  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.34) / 0.58) * 0.16 +
    clamp01((nearRatio - 0.18) / 0.7) * 0.28 +
    clamp01((gripLikeRatio - 0.24) / 0.66) * 0.16 +
    clamp01((pinchRatio - 0.18) / 0.72) * 0.1 +
    clamp01((downwardDelta - 0.02) / 0.34) * 0.18 +
    clamp01((yRange - 0.04) / 0.42) * 0.12,
  );

  const strictGate =
    oneHandRatio >= 0.28 &&
    nearRatio >= 0.18 &&
    gripLikeRatio >= 0.24 &&
    pinchRatio >= 0.2 &&
    downwardDelta >= 0.01;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Мужчина" ||
    current.confidence < 0.76 ||
    (["Женщина", "Солнце", "Дом", "Пока", "Привет", "Да", "Нет"].includes(current.label) &&
      current.confidence < 0.9);

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(Math.max(
    current.label === "Мужчина" ? current.confidence : 0,
    0.7 + detectorScore * 0.24,
  ));
  const secondaryLabel =
    current.label && current.label !== "Мужчина" ? current.label : "Женщина";
  const secondaryConfidence = clamp01(
    Math.min(confidence - 0.1, 0.78 - detectorScore * 0.18),
  );

  return {
    label: "Мужчина",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-muzhchina-forehead-detector",
    scores: [
      { label: "Мужчина", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "muzhchina_forehead_zone",
      detectorScore,
      side: dominantHand,
      nearRatio,
      pinchRatio,
      gripLikeRatio,
      proximityScore,
      lipProximityScore,
      xRange: 0,
      directionChanges: 0,
      oneHandRatio,
      downwardDelta,
    },
  };
}

function buildMuzhchinaLipPinchPrediction({
  frames = [],
  basePrediction = null,
} = {}) {
  if (!Array.isArray(frames) || frames.length < 8) {
    return null;
  }

  const current = getPredictionMetrics(basePrediction);
  const { dominantHand, handedness, handFrames, oneHandRatio } =
    getDominantHandFrameSet(frames);

  if (handFrames.length < Math.max(6, Math.round(frames.length * 0.44))) {
    return null;
  }

  let pinchCount = 0;
  let nearMouthCount = 0;
  let touchCount = 0;
  const touchSamples = [];
  const distanceSamples = [];

  for (const frame of handFrames) {
    const landmarks = frame.landmarks;
    const mouthTarget = getMouthTarget(frame?.face, frame?.pose);
    const noSecondHand = !(Array.isArray(frame?.opposite) && frame.opposite.length >= 21);

    if (!mouthTarget || !noSecondHand) {
      continue;
    }

    const indexTip = landmarks?.[8] ?? null;
    const thumbTip = landmarks?.[4] ?? null;

    if (!indexTip || !thumbTip) {
      continue;
    }

    const features = extractHandFeatures(landmarks, handedness);
    const scale = Math.max(Number(mouthTarget.scale ?? 0.1), 0.08);
    const pinch =
      getPointDistance(indexTip, thumbTip) / scale <= 0.58 ||
      Number(features?.thumbToIndexTip ?? 1) <= 0.52 ||
      Number(features?.fingerPinchLike ?? 0) >= 0.18;
    const mouthDistance = getPointDistance(indexTip, mouthTarget) / scale;
    const nearMouth = mouthDistance <= 0.92;

    if (pinch) {
      pinchCount += 1;
    }

    if (nearMouth) {
      nearMouthCount += 1;
    }

    if (pinch && nearMouth) {
      touchCount += 1;
      distanceSamples.push(mouthDistance);
      touchSamples.push({
        x: (Number(indexTip.x ?? 0) + Number(thumbTip.x ?? 0)) / 2,
        y: (Number(indexTip.y ?? 0) + Number(thumbTip.y ?? 0)) / 2,
        scale,
      });
    }
  }

  if (touchSamples.length < 4) {
    return null;
  }

  const pinchRatio = pinchCount / handFrames.length;
  const nearMouthRatio = nearMouthCount / handFrames.length;
  const touchRatio = touchCount / handFrames.length;
  const scaleAverage = Math.max(average(touchSamples.map((sample) => sample.scale)), 0.08);
  const xValues = touchSamples.map((sample) => sample.x);
  const yValues = touchSamples.map((sample) => sample.y);
  const xRangeNorm = (Math.max(...xValues) - Math.min(...xValues)) / scaleAverage;
  const yRangeNorm = (Math.max(...yValues) - Math.min(...yValues)) / scaleAverage;
  const distanceDirectionChanges = countDirectionChanges(distanceSamples, 0.02);
  const stableTouch = xRangeNorm <= 0.6 && yRangeNorm <= 0.7 && distanceDirectionChanges <= 1;

  const detectorScore = clamp01(
    clamp01((oneHandRatio - 0.36) / 0.54) * 0.18 +
      clamp01((pinchRatio - 0.18) / 0.68) * 0.28 +
      clamp01((nearMouthRatio - 0.18) / 0.68) * 0.24 +
      clamp01((touchRatio - 0.16) / 0.62) * 0.26 +
      (stableTouch ? 0.04 : 0),
  );

  const strictGate =
    oneHandRatio >= 0.42 &&
    pinchRatio >= 0.22 &&
    nearMouthRatio >= 0.2 &&
    touchRatio >= 0.18 &&
    stableTouch;

  if (!strictGate) {
    return null;
  }

  const canOverrideBasePrediction =
    !current.label ||
    current.label === "Мужчина" ||
    current.confidence < 0.8 ||
    (["Есть", "Спасибо", "Женщина", "Пить", "Пока", "Привет"].includes(current.label) &&
      current.confidence < 0.9);

  if (!canOverrideBasePrediction) {
    return null;
  }

  const confidence = clamp01(
    Math.max(current.label === "Мужчина" ? current.confidence : 0, 0.72 + detectorScore * 0.24),
  );
  const secondaryLabel = current.label && current.label !== "Мужчина" ? current.label : "Женщина";
  const secondaryConfidence = clamp01(Math.min(confidence - 0.1, 0.78 - detectorScore * 0.18));

  return {
    label: "Мужчина",
    confidence,
    recognitionLevel: "sign",
    sourceModel: "trajectory-muzhchina-lip-pinch-detector",
    scores: [
      { label: "Мужчина", confidence },
      { label: secondaryLabel, confidence: secondaryConfidence },
    ],
    detector: {
      type: "muzhchina_lip_pinch",
      detectorScore,
      oneHandRatio,
      pinchRatio,
      nearMouthRatio,
      touchRatio,
      xRangeNorm,
      yRangeNorm,
      distanceDirectionChanges,
      dominantHand,
    },
  };
}

function createEmptyCameraGuide() {
  return {
    mode: "privet",
    oneHandDone: false,
    waveDone: false,
    shoulderDone: false,
    headVisible: false,
    faceVisible: false,
    eyesVisible: false,
    noseVisible: false,
    lipsVisible: false,
    earsVisible: false,
    neckVisible: false,
    shoulderVisible: false,
    elbowVisible: false,
    handVisible: false,
    fingersVisible: false,
    poseDetected: false,
    handsDetectedCount: 0,
    waveScore: 0,
    readyCount: 0,
  };
}

function normalizeCameraGuide(guide) {
  if (!guide || typeof guide !== "object") {
    return createEmptyCameraGuide();
  }

  return {
    mode: String(guide.mode ?? "privet"),
    oneHandDone: Boolean(guide.oneHandDone),
    waveDone: Boolean(guide.waveDone),
    shoulderDone: Boolean(guide.shoulderDone),
    headVisible: Boolean(guide.headVisible),
    faceVisible: Boolean(guide.faceVisible),
    eyesVisible: Boolean(guide.eyesVisible),
    noseVisible: Boolean(guide.noseVisible),
    lipsVisible: Boolean(guide.lipsVisible),
    earsVisible: Boolean(guide.earsVisible),
    neckVisible: Boolean(guide.neckVisible),
    shoulderVisible: Boolean(guide.shoulderVisible),
    elbowVisible: Boolean(guide.elbowVisible),
    handVisible: Boolean(guide.handVisible),
    fingersVisible: Boolean(guide.fingersVisible),
    poseDetected: Boolean(guide.poseDetected),
    handsDetectedCount: Number(guide.handsDetectedCount ?? 0),
    waveScore: Number(guide.waveScore ?? 0),
    readyCount: Number(guide.readyCount ?? 0),
  };
}

function getPhraseChecklistDoneStates(phraseId, guide, isPhraseMatched) {
  const normalizedId = String(phraseId ?? "").trim().toLowerCase();
  const handsDetectedCount = Number(guide?.handsDetectedCount ?? 0);
  const hasOneHand = handsDetectedCount >= 1 || Boolean(guide?.handVisible);
  const hasTwoHands = handsDetectedCount >= 2;
  const faceVisible = Boolean(guide?.faceVisible);
  const shoulderVisible = Boolean(guide?.shoulderVisible) || Boolean(guide?.poseDetected);
  const phraseMatched = Boolean(isPhraseMatched);

  switch (normalizedId) {
    case "privet":
      return [
        Boolean(guide?.oneHandDone),
        Boolean(guide?.waveDone),
        Boolean(guide?.shoulderDone),
      ];
    case "dom":
    case "druzhba":
    case "lyubov":
    case "bolshoy":
      return [hasTwoHands, shoulderVisible, phraseMatched];
    case "muzhchina":
    case "zhenshchina":
      return [hasOneHand, faceVisible, phraseMatched];
    case "ya":
    case "ty":
    case "horosho":
    case "stop":
    case "est":
    case "pit":
    case "spat":
    case "idti":
    case "bezhat":
    case "dumat":
    case "poka":
    case "solntse":
    case "da":
    case "net":
    default:
      return [hasOneHand, shoulderVisible, phraseMatched];
  }
}

export default function App({ uiMode = "admin" }) {
  const isUserUi = uiMode === "user";
  const [locale, setLocale] = useState("ru");
  const [theme, setTheme] = useState("dark");
  const [bootstrapPhrases, setBootstrapPhrases] = useState([]);
  const [bootstrapUserEmail, setBootstrapUserEmail] = useState("");
  const [autoSpeakEnabled, setAutoSpeakEnabled] = useState(true);
  const [activeTab, setActiveTab] = useState(isUserUi ? "camera" : "home");
  const [cameraStartSignal, setCameraStartSignal] = useState(0);
  const [cameraRecognitionLevel, setCameraRecognitionLevel] = useState(
    isUserUi ? "sign" : "alphabet",
  );
  const [cameraFacingMode, setCameraFacingMode] = useState("user");
  const [isCameraPreviewMirrored] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [voiceStatusKey, setVoiceStatusKey] = useState(() =>
    getSpeechRecognitionApi() ? "ready" : "unsupported",
  );
  const [livePrediction, setLivePrediction] = useState(null);
  const [recognizedWord, setRecognizedWord] = useState("");
  const [liveError, setLiveError] = useState("");
  const [isLivePredicting, setIsLivePredicting] = useState(false);
  const [cameraGuide, setCameraGuide] = useState(() => createEmptyCameraGuide());
  const [openedPhraseGuideId, setOpenedPhraseGuideId] = useState(null);
  const gestureBridgeLifecycleRef = useRef(false);
  const recognitionRef = useRef(null);
  const transcriptRef = useRef("");
  const endStatusKeyRef = useRef(getSpeechRecognitionApi() ? "ready" : "unsupported");
  const liveRequestRef = useRef(0);
  const livePredictInFlightRef = useRef(false);
  const liveLastRequestAtRef = useRef(0);
  const livePredictionWindowRef = useRef([]);
  const liveStableLabelRef = useRef("");
  const lastStrongSignPredictionRef = useRef({
    prediction: null,
    at: 0,
  });
  const lastSpokenPredictionRef = useRef({
    key: "",
    spokenAt: 0,
  });
  const lastAlphabetAppendRef = useRef("");
  const phraseModelUnavailableRef = useRef(false);
  const lastSyncedLocaleRef = useRef("ru");
  const lastSyncedAutoSpeakRef = useRef(null);
  const bootstrapReadyRef = useRef(false);
  const {
    connectionState: gestureBridgeConnectionState,
    connected: gestureBridgeConnected,
    workerReady: gestureBridgeWorkerReady,
    status: gestureBridgeStatus,
    prediction: gestureBridgePrediction,
    diagnostics: gestureBridgeDiagnostics,
    error: gestureBridgeError,
    start: startGestureBridgeSession,
    stop: stopGestureBridgeSession,
    reset: resetGestureBridgeSession,
    sendFrame: sendGestureBridgeFrame,
    clearLocalState: clearGestureBridgeState,
  } = useGestureBridge();

  const t = copy[locale] ?? copy.ru;
  const voiceStatusText = getVoiceStatusText(t, voiceStatusKey);
  const fallbackCameraPhraseLibraryItems = [
    {
      id: "privet",
      label: t.camera.phraseLibraryItems.privet,
      aliases: ["Привет", "Hello"],
      guide: t.camera.phraseGuideItems.privet,
      checklist: t.camera.phraseChecklistItems.privet,
    },
    {
      id: "poka",
      label: t.camera.phraseLibraryItems.poka,
      aliases: ["Пока", "Bye"],
      guide: t.camera.phraseGuideItems.poka,
      checklist: t.camera.phraseChecklistItems.poka,
    },
    {
      id: "ya",
      label: t.camera.phraseLibraryItems.ya,
      aliases: ["Я", "I", "Me"],
      guide: t.camera.phraseGuideItems.ya,
      checklist: t.camera.phraseChecklistItems.ya,
    },
    {
      id: "muzhchina",
      label: t.camera.phraseLibraryItems.muzhchina,
      aliases: ["Мужчина", "Man"],
      guide: t.camera.phraseGuideItems.muzhchina,
      checklist: t.camera.phraseChecklistItems.muzhchina,
    },
    {
      id: "zhenshchina",
      label: t.camera.phraseLibraryItems.zhenshchina,
      aliases: ["Женщина", "Woman"],
      guide: t.camera.phraseGuideItems.zhenshchina,
      checklist: t.camera.phraseChecklistItems.zhenshchina,
    },
    {
      id: "spasibo",
      label: t.camera.phraseLibraryItems.spasibo,
      aliases: ["Спасибо", "Thank you"],
      guide: t.camera.phraseGuideItems.spasibo,
      checklist: t.camera.phraseChecklistItems.spasibo,
    },
  ];
  const cameraPhraseLibraryItems = bootstrapPhrases.length
    ? buildPhraseLibraryItemsFromServer(bootstrapPhrases, t)
    : fallbackCameraPhraseLibraryItems;
  const fallbackCameraSignLibraryItems = [];
  const cameraSignLibraryItems = fallbackCameraSignLibraryItems;
  const cameraSideLibraryItems =
    cameraRecognitionLevel === "sign"
      ? (cameraSignLibraryItems.length ? cameraSignLibraryItems : fallbackCameraSignLibraryItems).map((item) => ({
        ...item,
        zoneHint: FACE_AREA_SIGN_IDS.has(item.id) ? t.camera.faceZoneHint : "",
      }))
      : cameraPhraseLibraryItems;
  const cameraVisionItems = [
    { id: "head", label: t.camera.guideVisionItems.head, done: cameraGuide.headVisible },
    { id: "face", label: t.camera.guideVisionItems.face, done: cameraGuide.faceVisible },
    { id: "eyes", label: t.camera.guideVisionItems.eyes, done: cameraGuide.eyesVisible },
    { id: "nose", label: t.camera.guideVisionItems.nose, done: cameraGuide.noseVisible },
    { id: "lips", label: t.camera.guideVisionItems.lips, done: cameraGuide.lipsVisible },
    { id: "ears", label: t.camera.guideVisionItems.ears, done: cameraGuide.earsVisible },
    { id: "neck", label: t.camera.guideVisionItems.neck, done: cameraGuide.neckVisible },
    { id: "shoulder", label: t.camera.guideVisionItems.shoulder, done: cameraGuide.shoulderVisible },
    { id: "elbow", label: t.camera.guideVisionItems.elbow, done: cameraGuide.elbowVisible },
    { id: "hand", label: t.camera.guideVisionItems.hand, done: cameraGuide.handVisible },
    { id: "fingers", label: t.camera.guideVisionItems.fingers, done: cameraGuide.fingersVisible },
  ];
  const isGitHubPages =
    typeof window !== "undefined" &&
    window.location.hostname.endsWith("github.io");
  const rawSignBridgePrediction =
    gestureBridgePrediction?.label_id && gestureBridgePrediction.label_id !== "none"
      ? {
        label: resolveGestureLabelRu(
          gestureBridgePrediction.label_id,
          gestureBridgePrediction.label_ru,
        ),
        confidence: gestureBridgePrediction.confidence,
      }
      : null;
  const signBridgePrediction =
    rawSignBridgePrediction && ALLOWED_WORD_LABELS.has(String(rawSignBridgePrediction.label ?? "").trim())
      ? rawSignBridgePrediction
      : null;
  const activeCameraPrediction =
    cameraRecognitionLevel === "sign"
      ? (livePrediction ?? signBridgePrediction)
      : livePrediction;
  const activeCameraError =
    cameraRecognitionLevel === "sign" && !isGitHubPages
      ? gestureBridgeError
      : liveError;
  const signTrackingOk = Boolean(
    gestureBridgePrediction?.debug?.tracking_ok ??
      gestureBridgeStatus?.tracking_ok ??
      gestureBridgeStatus?.trackingOk,
  );
  const signBufferSize = Number(
    gestureBridgeStatus?.buffer_size ?? gestureBridgeStatus?.bufferSize ?? 0,
  );
  const allowedBridgeSignals = (signals = []) =>
    signals
      .map((item) => {
        const label = resolveGestureLabelRu(item.label_id, item.label_ru);
        const normalizedLabel = String(label ?? "").trim();

        if (!ALLOWED_WORD_LABELS.has(normalizedLabel)) {
          return null;
        }

        return { ...item, resolvedLabel: normalizedLabel };
      })
      .filter(Boolean);
  const signTopRuleSignals = Array.isArray(gestureBridgePrediction?.top_rules)
    ? allowedBridgeSignals(gestureBridgePrediction.top_rules).slice(0, 3)
    : [];
  const signTopModelSignals = Array.isArray(gestureBridgePrediction?.top_model)
    ? allowedBridgeSignals(gestureBridgePrediction.top_model).slice(0, 3)
    : [];
  const signTopFinalSignals = Array.isArray(gestureBridgePrediction?.top_final)
    ? allowedBridgeSignals(gestureBridgePrediction.top_final).slice(0, 3)
    : [];
  const uiModeLabel = isUserUi ? "User mode" : "Admin mode";
  const signUiState =
    cameraStartSignal <= 0
      ? "camera_off"
      : activeCameraError
        ? "error"
        : !gestureBridgeConnected
          ? "connecting_backend"
          : !gestureBridgeWorkerReady
            ? "python_starting"
            : !signTrackingOk && signBufferSize > 0
              ? "tracking_lost"
              : gestureBridgeStatus?.frameInFlight
                ? "predicting"
                : "ready";
  const livePredictionLabel = String(activeCameraPrediction?.label ?? "").trim().toLowerCase();
  const showExternalCameraPhraseLibrary =
    !isUserUi &&
    activeTab === "camera" &&
    (
      cameraRecognitionLevel === "alphabet" ||
      cameraRecognitionLevel === "phrase" ||
      cameraRecognitionLevel === "sign"
    );
  const showExternalCameraVisionPanel =
    !isUserUi && activeTab === "camera" && cameraRecognitionLevel === "phrase";
  const visibleVisionCount = cameraVisionItems.filter((item) => item.done).length;
  const openedPhraseGuide = cameraSideLibraryItems.find(
    (item) => item.id === openedPhraseGuideId,
  );
  const focusedPhraseLabel =
    cameraRecognitionLevel !== "alphabet"
      ? (openedPhraseGuide?.aliases?.[0] ?? PHRASE_LABEL_BY_ID[openedPhraseGuideId] ?? "")
      : "";
  const useNativeMobileShell = isUserUi;

  function switchUiMode(nextMode) {
    const normalizedMode = nextMode === "user" ? "user" : "admin";
    const nextUrl = new URL(window.location.href);

    if (normalizedMode === "user") {
      nextUrl.searchParams.set("ui", "user");
    } else {
      nextUrl.searchParams.set("ui", "admin");
    }

    window.location.href = nextUrl.toString();
  }

  function renderUiModeSwitch(compact = false) {
    return (
      <div className={`ui-mode-switch ${compact ? "compact" : ""}`}>
        <span className="ui-mode-label">{uiModeLabel}</span>
        <div className="ui-mode-actions">
          <button
            className={`ui-mode-button ${!isUserUi ? "active" : ""}`}
            type="button"
            onClick={() => switchUiMode("admin")}
          >
            Admin
          </button>
          <button
            className={`ui-mode-button ${isUserUi ? "active" : ""}`}
            type="button"
            onClick={() => switchUiMode("user")}
          >
            User
          </button>
        </div>
      </div>
    );
  }

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const data = await fetchBootstrapData();

        if (cancelled) {
          return;
        }

        const nextLocale = data?.settings?.uiLanguage === "en" ? "en" : "ru";
        const nextAutoSpeakEnabled = Boolean(data?.settings?.autoSpeakEnabled ?? true);

        setBootstrapPhrases(Array.isArray(data?.phrases) ? data.phrases : []);
        setBootstrapUserEmail(data?.defaultUser?.email ?? "");
        lastSyncedLocaleRef.current = nextLocale;
        setLocale(nextLocale);
        lastSyncedAutoSpeakRef.current = nextAutoSpeakEnabled;
        setAutoSpeakEnabled(nextAutoSpeakEnabled);
      } catch {
        // Fall back to the built-in copy if bootstrap is unavailable.
      } finally {
        if (!cancelled) {
          bootstrapReadyRef.current = true;
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!bootstrapReadyRef.current || !bootstrapUserEmail) {
      return;
    }

    if (locale === lastSyncedLocaleRef.current) {
      return;
    }

    lastSyncedLocaleRef.current = locale;
    updateSettings({
      userEmail: bootstrapUserEmail,
      uiLanguage: locale,
    }).catch(() => {});
  }, [bootstrapUserEmail, locale]);

  useEffect(() => {
    if (!bootstrapReadyRef.current || !bootstrapUserEmail) {
      return;
    }

    if (autoSpeakEnabled === lastSyncedAutoSpeakRef.current) {
      return;
    }

    lastSyncedAutoSpeakRef.current = autoSpeakEnabled;
    updateSettings({
      userEmail: bootstrapUserEmail,
      autoSpeakEnabled,
    }).catch(() => {});
  }, [autoSpeakEnabled, bootstrapUserEmail]);

  function renderCameraPredictionPanel() {
    if (activeTab !== "camera") {
      return null;
    }

    const showNonePrediction =
      cameraRecognitionLevel === "sign" &&
      cameraStartSignal > 0 &&
      gestureBridgeWorkerReady &&
      gestureBridgePrediction?.label_id === "none";
    const signStateLabelMap = {
      camera_off: "Камера выключена",
      connecting_backend: "Подключаем backend",
      python_starting: "Python worker запускается",
      tracking_lost: "Трекинг потерян",
      predicting: "Идёт анализ",
      ready: "Система готова",
      error: "Ошибка",
    };

    return (
      <>
        <div className="camera-prediction-card">
          <span className="transcript-kicker">Live</span>
          {activeCameraError ? (
            <p className="camera-prediction-error">{activeCameraError}</p>
          ) : activeCameraPrediction?.label ? (
            <div className="camera-prediction-value">
              <strong>{activeCameraPrediction.label}</strong>
              <span>{Math.round((activeCameraPrediction.confidence ?? 0) * 100)}%</span>
            </div>
          ) : showNonePrediction ? (
            <p className="camera-prediction-empty">Нет жеста</p>
          ) : (
            <p className="camera-prediction-empty">
              {cameraStartSignal > 0
                ? cameraRecognitionLevel === "sign"
                  ? signStateLabelMap[signUiState] ?? t.camera.waiting
                  : isLivePredicting
                  ? t.camera.analyzing
                  : cameraRecognitionLevel === "alphabet"
                    ? t.camera.waitingAlphabet
                    : t.camera.waiting
                : t.camera.idle}
            </p>
          )}
        </div>

        {cameraRecognitionLevel === "sign" && !isUserUi && !isGitHubPages ? (
          <div className="camera-guide-card camera-gesture-debug-card">
            <div className="camera-guide-header">
              <span className="camera-guide-section-title">Gesture bridge</span>
              <button
                className="camera-toggle-button"
                type="button"
                onClick={() => {
                  resetGestureBridgeSession()
                    .then(() => {
                      clearGestureBridgeState();
                    })
                    .catch(() => {});
                }}
              >
                Сброс
              </button>
            </div>
            <div className="camera-gesture-debug-grid">
              <div className="camera-gesture-debug-item">
                <span>UI state</span>
                <strong>{signStateLabelMap[signUiState] ?? signUiState}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Backend</span>
                <strong>{gestureBridgeConnected ? "connected" : gestureBridgeConnectionState}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Python</span>
                <strong>{gestureBridgeWorkerReady ? "ready" : "starting"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Tracking</span>
                <strong>{signTrackingOk ? "ok" : "weak"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Send FPS</span>
                <strong>{Math.round(gestureBridgeDiagnostics?.sendFps ?? 0)}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Latency</span>
                <strong>
                  {gestureBridgeDiagnostics?.lastRoundtripMs
                    ? `${Math.round(gestureBridgeDiagnostics.lastRoundtripMs)} ms`
                    : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Ping</span>
                <strong>
                  {gestureBridgeDiagnostics?.lastPingLatencyMs
                    ? `${Math.round(gestureBridgeDiagnostics.lastPingLatencyMs)} ms`
                    : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Буфер</span>
                <strong>{signBufferSize}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Busy</span>
                <strong>{gestureBridgeStatus?.frameInFlight ? "yes" : "no"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Рук</span>
                <strong>{gestureBridgePrediction?.debug?.hands_count ?? gestureBridgeStatus?.hands_count ?? 0}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Зона</span>
                <strong>{gestureBridgePrediction?.debug?.dominant_zone ?? gestureBridgeStatus?.dominant_zone ?? "—"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Движение</span>
                <strong>{gestureBridgePrediction?.debug?.movement_type ?? gestureBridgeStatus?.movement_type ?? "—"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Повторы</span>
                <strong>{gestureBridgePrediction?.debug?.repeat_count ?? gestureBridgeStatus?.repeat_count ?? 0}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Top label</span>
                <strong>
                  {gestureBridgePrediction?.top_label_id
                    ? (() => {
                      const topLabel = resolveGestureLabelRu(
                        gestureBridgePrediction.top_label_id,
                        gestureBridgePrediction?.top_label_ru ?? gestureBridgePrediction?.label_ru,
                      );
                      const normalizedTopLabel = String(topLabel ?? "").trim();
                      return ALLOWED_WORD_LABELS.has(normalizedTopLabel) ? normalizedTopLabel : "—";
                    })()
                    : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Top confidence</span>
                <strong>
                  {gestureBridgePrediction?.top_confidence
                    ? `${Math.round(gestureBridgePrediction.top_confidence * 100)}%`
                    : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>None reason</span>
                <strong>{gestureBridgePrediction?.debug?.none_reason ?? gestureBridgeStatus?.none_reason ?? "—"}</strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Drops</span>
                <strong>
                  {Number(gestureBridgeDiagnostics?.framesDroppedBusyLocal ?? 0) +
                    Number(gestureBridgeDiagnostics?.framesDroppedSocket ?? 0) +
                    Number(gestureBridgeDiagnostics?.framesDroppedHidden ?? 0) +
                    Number(gestureBridgeDiagnostics?.framesDroppedOversizeLocal ?? 0) +
                    Number(gestureBridgeDiagnostics?.framesDroppedByBackend ?? 0)}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Processing</span>
                <strong>
                  {gestureBridgePrediction?.debug?.processing_ms
                    ? `${Math.round(gestureBridgePrediction.debug.processing_ms)} ms`
                    : gestureBridgeStatus?.processing_ms
                      ? `${Math.round(gestureBridgeStatus.processing_ms)} ms`
                      : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Inference</span>
                <strong>
                  {gestureBridgePrediction?.debug?.inference_ms
                    ? `${Math.round(gestureBridgePrediction.debug.inference_ms)} ms`
                    : "—"}
                </strong>
              </div>
              <div className="camera-gesture-debug-item">
                <span>Drop reason</span>
                <strong>{gestureBridgeDiagnostics?.lastFrameDropReason || "—"}</strong>
              </div>
            </div>

            <div className="camera-gesture-signal-block">
              <span className="camera-guide-section-title">Top final</span>
              <div className="camera-gesture-signal-list">
                {signTopFinalSignals.length ? (
                  signTopFinalSignals.map((item) => (
                    <div key={`${item.label_id}-${item.score}`} className="camera-gesture-signal-pill">
                      <strong>{item.resolvedLabel}</strong>
                      <span>{Math.round(item.score * 100)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="camera-prediction-empty">Пока нет финальных сигналов</p>
                )}
              </div>
            </div>

            <div className="camera-gesture-signal-block">
              <span className="camera-guide-section-title">Top rules</span>
              <div className="camera-gesture-signal-list">
                {signTopRuleSignals.length ? (
                  signTopRuleSignals.map((item) => (
                    <div key={`${item.kind}-${item.label_id}-${item.score}`} className="camera-gesture-signal-pill">
                      <strong>{item.kind}:{item.resolvedLabel}</strong>
                      <span>{Math.round(item.score * 100)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="camera-prediction-empty">Правила ещё не накопили сигнал</p>
                )}
              </div>
            </div>

            <div className="camera-gesture-signal-block">
              <span className="camera-guide-section-title">Top model</span>
              <div className="camera-gesture-signal-list">
                {signTopModelSignals.length ? (
                  signTopModelSignals.map((item) => (
                    <div key={`${item.label_id}-${item.score}`} className="camera-gesture-signal-pill">
                      <strong>{item.resolvedLabel}</strong>
                      <span>{Math.round(item.score * 100)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="camera-prediction-empty">Model scores пока пустые</p>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {cameraRecognitionLevel === "alphabet" ? (
          <>
            <div className="camera-prediction-card camera-word-card">
              <span className="transcript-kicker">
                {isUserUi ? t.camera.wordTitle : t.camera.lettersTitle}
              </span>
              {recognizedWord ? (
                <p className="camera-word-value">{recognizedWord}</p>
              ) : (
                <p className="camera-prediction-empty">
                  {isUserUi ? t.camera.wordPlaceholder : t.camera.lettersPlaceholder}
                </p>
              )}

              {isUserUi ? (
                <div className="camera-word-actions">
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={() => {
                      setRecognizedWord((previousValue) =>
                        previousValue.length > 0 ? previousValue.slice(0, -1) : "",
                      );
                      lastAlphabetAppendRef.current = "";
                    }}
                  >
                    {t.camera.wordBackspace}
                  </button>
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={speakRecognizedWord}
                    disabled={!recognizedWord}
                  >
                    {t.camera.wordSpeak}
                  </button>
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={() => {
                      setRecognizedWord("");
                      lastAlphabetAppendRef.current = "";
                    }}
                  >
                    {t.camera.wordClear}
                  </button>
                </div>
              ) : null}
            </div>

            {!isUserUi ? (
              <div className="camera-prediction-card camera-word-card">
                <span className="transcript-kicker">{t.camera.wordTitle}</span>
                {recognizedWord ? (
                  <p className="camera-word-value">{recognizedWord}</p>
                ) : (
                  <p className="camera-prediction-empty">{t.camera.wordPlaceholder}</p>
                )}
                <div className="camera-word-actions">
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={() => {
                      setRecognizedWord((previousValue) =>
                        previousValue.length > 0 ? previousValue.slice(0, -1) : "",
                      );
                      lastAlphabetAppendRef.current = "";
                    }}
                  >
                    {t.camera.wordBackspace}
                  </button>
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={speakRecognizedWord}
                    disabled={!recognizedWord}
                  >
                    {t.camera.wordSpeak}
                  </button>
                  <button
                    className="camera-toggle-button"
                    type="button"
                    onClick={() => {
                      setRecognizedWord("");
                      lastAlphabetAppendRef.current = "";
                    }}
                  >
                    {t.camera.wordClear}
                  </button>
                </div>
              </div>
            ) : null}
          </>
        ) : null}
      </>
    );
  }

  function renderCameraPhraseLibraryPanel() {
    if (cameraRecognitionLevel === "alphabet") {
      return (
        <div className="camera-side-panel">
          {renderCameraPredictionPanel()}
        </div>
      );
    }

    if (cameraRecognitionLevel === "sign" && cameraSideLibraryItems.length === 0) {
      return (
        <div className="camera-side-panel">
          {renderCameraPredictionPanel()}
        </div>
      );
    }

    const openedPhraseMatched = Boolean(
      openedPhraseGuide?.aliases?.some(
        (alias) => String(alias).toLowerCase() === livePredictionLabel,
      ),
    );
    const openedPhraseDoneStates = openedPhraseGuide
      ? getPhraseChecklistDoneStates(openedPhraseGuide.id, cameraGuide, openedPhraseMatched)
      : [false, false, false];
    const openedPhraseChecklistItems = (openedPhraseGuide?.checklist || [])
      .slice(0, 3)
      .map((label, index) => ({
        id: `${openedPhraseGuide?.id || "phrase"}-check-${index + 1}`,
        label,
        detail: openedPhraseGuide?.guide?.steps?.[index] || "",
        done: Boolean(openedPhraseDoneStates[index]),
      }));
    const openedPhraseProgress = openedPhraseChecklistItems.filter((item) => item.done).length;

    return (
      <div className="camera-side-panel camera-phrase-panel-stack">
        {renderCameraPredictionPanel()}

        <div className="camera-guide-card camera-phrase-library-card">
          <div className="camera-phrase-grid">
            {cameraSideLibraryItems.map((item) => {
              const isPredicted = item.aliases.some(
                (alias) => alias.toLowerCase() === livePredictionLabel,
              );
              const isOpened = openedPhraseGuide?.id === item.id;

              return (
                <button
                  key={item.id}
                  className={`camera-phrase-button ${isPredicted ? "predicted" : ""} ${isOpened ? "opened" : ""}`}
                  type="button"
                  onClick={() => setOpenedPhraseGuideId(item.id)}
                >
                  <span className="camera-phrase-button-label">{item.label}</span>
                  {item.zoneHint ? (
                    <span className="camera-phrase-button-hint">{item.zoneHint}</span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </div>

        {openedPhraseGuide ? (
          <div
            className="camera-guide-card camera-phrase-guide-window"
            role="dialog"
            aria-label={openedPhraseGuide.guide?.title || openedPhraseGuide.label}
          >
            <div className="camera-guide-header">
              <div className="camera-guide-copy">
                <span className="camera-guide-section-title">{t.camera.phraseGuideLiveTitle}</span>
                <strong>{openedPhraseGuide.guide?.title || openedPhraseGuide.label}</strong>
                <p>{openedPhraseGuide.guide?.text}</p>
                {openedPhraseGuide.zoneHint ? (
                  <p className="camera-guide-note">{openedPhraseGuide.zoneHint}</p>
                ) : null}
              </div>
              <button
                className="camera-phrase-guide-close"
                type="button"
                onClick={() => setOpenedPhraseGuideId(null)}
              >
                {t.camera.phraseGuideClose}
              </button>
            </div>

            <div className="camera-guide-header">
              <span className="camera-guide-section-title">{t.camera.phraseGuideLiveTitle}</span>
              <span className="camera-guide-progress">{openedPhraseProgress}/3</span>
            </div>
            <div className="camera-guide-list">
              {openedPhraseChecklistItems.map((item) => (
                <div
                  key={item.id}
                  className={`camera-guide-item ${item.done ? "done" : ""}`}
                >
                  <span className="camera-guide-check" aria-hidden="true">
                    {item.done ? "✓" : ""}
                  </span>
                  <span className="camera-guide-item-copy">
                    <strong>{item.label}</strong>
                    {item.detail ? <small>{item.detail}</small> : null}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  function renderCameraVisionPanel() {
    return (
      <div className="camera-guide-card">
        <div className="camera-guide-header">
          <span className="camera-guide-section-title">{t.camera.guideVisionTitle}</span>
          <span className="camera-guide-progress">{visibleVisionCount}/{cameraVisionItems.length}</span>
        </div>
        <div className="camera-guide-list">
          {cameraVisionItems.map((item) => (
            <div
              key={item.id}
              className={`camera-guide-item ${item.done ? "done" : ""}`}
            >
              <span className="camera-guide-check" aria-hidden="true">
                {item.done ? "✓" : ""}
              </span>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  useEffect(() => {
    document.documentElement.lang = t.htmlLang;
    document.documentElement.dataset.appTheme = theme;
    document.title = "Silent conversation";
  }, [t.htmlLang, theme]);

  useEffect(() => {
    if (recognitionRef.current) {
      recognitionRef.current.lang = copy[locale].speechLocale;
    }
  }, [locale]);

  useEffect(() => {
    if (cameraRecognitionLevel !== "phrase") {
      return undefined;
    }

    let cancelled = false;

    (async () => {
      try {
        await fetchLatestModel("phrase", { profile: "fast" });
        if (!cancelled) {
          phraseModelUnavailableRef.current = false;
        }
      } catch (error) {
        if (!cancelled && isModelNotTrainedError(error)) {
          phraseModelUnavailableRef.current = true;
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [cameraRecognitionLevel]);

  useEffect(() => {
    if (!showExternalCameraPhraseLibrary && openedPhraseGuideId) {
      setOpenedPhraseGuideId(null);
    }
  }, [showExternalCameraPhraseLibrary, openedPhraseGuideId]);

  useEffect(() => {
    if (isGitHubPages) {
      if (gestureBridgeLifecycleRef.current) {
        gestureBridgeLifecycleRef.current = false;
        stopGestureBridgeSession().catch(() => {});
        clearGestureBridgeState();
      }
      return;
    }

    const shouldRunGestureBridge =
      activeTab === "camera" &&
      cameraRecognitionLevel === "sign" &&
      cameraStartSignal > 0;

    if (shouldRunGestureBridge === gestureBridgeLifecycleRef.current) {
      return;
    }

    gestureBridgeLifecycleRef.current = shouldRunGestureBridge;

    if (shouldRunGestureBridge) {
      startGestureBridgeSession().catch(() => {});
      return;
    }

    stopGestureBridgeSession().catch(() => {});
    clearGestureBridgeState();
  }, [activeTab, cameraRecognitionLevel, cameraStartSignal, clearGestureBridgeState, isGitHubPages, startGestureBridgeSession, stopGestureBridgeSession]);

  useEffect(() => {
    const SpeechRecognitionApi = getSpeechRecognitionApi();

    if (!SpeechRecognitionApi) {
      return undefined;
    }

    const recognition = new SpeechRecognitionApi();
    recognition.lang = copy.ru.speechLocale;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      setIsRecording(true);
      setVoiceStatusKey("listening");
      endStatusKeyRef.current = "finishedSaved";
    };
    recognition.onresult = (event) => {
      let nextFinal = transcriptRef.current;
      let nextInterim = "";

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const phrase = event.results[index][0].transcript.trim();

        if (!phrase) {
          continue;
        }

        if (event.results[index].isFinal) {
          nextFinal = `${nextFinal} ${phrase}`.trim();
        } else {
          nextInterim = `${nextInterim} ${phrase}`.trim();
        }
      }

      transcriptRef.current = nextFinal;
      setTranscript(nextFinal);
      setInterimTranscript(nextInterim);

      if (nextFinal || nextInterim) {
        setVoiceStatusKey("transforming");
      }
    };
    recognition.onerror = ({ error }) => {
      setIsRecording(false);
      setInterimTranscript("");
      setVoiceStatusKey(`errors.${error}`);
    };
    recognition.onend = () => {
      setIsRecording(false);
      setInterimTranscript("");
      setVoiceStatusKey(endStatusKeyRef.current);
    };

    recognitionRef.current = recognition;

    return () => {
      try {
        recognition.stop();
      } catch {
        // Ignore cleanup errors.
      }
      recognitionRef.current = null;
    };
  }, []);

  function stopVoiceRecognition(finalStatusKey) {
    if (!recognitionRef.current) {
      setVoiceStatusKey(finalStatusKey);
      return;
    }

    endStatusKeyRef.current = finalStatusKey;
    setVoiceStatusKey("stopping");

    try {
      recognitionRef.current.stop();
    } catch {
      setIsRecording(false);
      setInterimTranscript("");
      setVoiceStatusKey(finalStatusKey);
    }
  }

  function handleTabChange(tabId) {
    if (tabId !== "voice" && isRecording) {
      stopVoiceRecognition(transcriptRef.current ? "finishedSaved" : "stopped");
    }

    if (tabId !== "camera" && activeTab === "camera") {
      setCameraStartSignal(0);
      setLivePrediction(null);
      setLiveError("");
      setCameraGuide(createEmptyCameraGuide());
      livePredictionWindowRef.current = [];
      liveStableLabelRef.current = "";
      lastStrongSignPredictionRef.current = {
        prediction: null,
        at: 0,
      };
      lastSpokenPredictionRef.current = { key: "", spokenAt: 0 };
      lastAlphabetAppendRef.current = "";
      clearGestureBridgeState();
    }

    setActiveTab(tabId);
  }

  function resetLiveCameraPrediction() {
    setLivePrediction(null);
    setLiveError("");
    setIsLivePredicting(false);
    setCameraGuide(createEmptyCameraGuide());
    livePredictInFlightRef.current = false;
    livePredictionWindowRef.current = [];
    liveStableLabelRef.current = "";
    lastStrongSignPredictionRef.current = {
      prediction: null,
      at: 0,
    };
    lastSpokenPredictionRef.current = { key: "", spokenAt: 0 };
    lastAlphabetAppendRef.current = "";
    liveRequestRef.current += 1;
    clearGestureBridgeState();
  }

  function appendAlphabetToWord(label) {
    const normalizedLabel = String(label ?? "").trim();

    if (!normalizedLabel) {
      return;
    }

    if (lastAlphabetAppendRef.current === normalizedLabel) {
      return;
    }

    setRecognizedWord((previousValue) => `${previousValue}${normalizedLabel}`);
    lastAlphabetAppendRef.current = normalizedLabel;
  }

  function speakRecognizedWord() {
    const normalizedWord = String(recognizedWord ?? "").trim().replace(/\s+/g, " ");

    if (!normalizedWord) {
      return;
    }

    speakWithSettings(
      normalizedWord,
      { uiLanguage: locale },
      { recognitionLevel: "alphabet_word" },
    );
  }

  function speakLivePrediction(label, recognitionLevel) {
    const normalizedLabel = String(label ?? "").trim();
    const normalizedRecognitionLevel = String(recognitionLevel ?? "").trim().toLowerCase();

    if (!normalizedLabel) {
      return;
    }

    if (!autoSpeakEnabled) {
      return;
    }

    const now = Date.now();
    const speechKey = `${normalizedRecognitionLevel}:${normalizedLabel}`;
    const isSameAsLast = lastSpokenPredictionRef.current.key === speechKey;
    const tooSoon = now - lastSpokenPredictionRef.current.spokenAt < LIVE_SPEECH_MIN_INTERVAL_MS;

    if (isSameAsLast || tooSoon) {
      return;
    }

    const didSpeak = speakWithSettings(
      normalizedLabel,
      { uiLanguage: locale },
      { recognitionLevel: normalizedRecognitionLevel },
    );

    if (didSpeak) {
      lastSpokenPredictionRef.current = { key: speechKey, spokenAt: now };
    }
  }

  function handleCameraRecognitionChange(nextLevel) {
    const normalizedLevel = nextLevel === "phrase" ? "sign" : nextLevel;

    if (normalizedLevel === cameraRecognitionLevel) {
      return;
    }

    setCameraRecognitionLevel(normalizedLevel);
    resetLiveCameraPrediction();
  }

  function handleCameraFacingToggle() {
    setCameraFacingMode((value) => (value === "user" ? "environment" : "user"));
    resetLiveCameraPrediction();
  }

  function handleCameraToggle() {
    if (cameraRecognitionLevel === "phrase") {
      phraseModelUnavailableRef.current = false;
    }
    if (cameraRecognitionLevel === "sign") {
      resetGestureBridgeSession().catch(() => {});
    }
    resetLiveCameraPrediction();
    setCameraStartSignal((value) => (value > 0 ? 0 : Date.now()));
  }

  function handleCameraPredictionChange(payload) {
    const nextGuide = normalizeCameraGuide(payload?.guide);
    const shouldClearLivePrediction =
      payload &&
      (
        payload.gesture === "Жест не найден" ||
        Number(payload.windowSize ?? 0) === 0 ||
        payload.handVisibleEnough === false
      );

    if (shouldClearLivePrediction) {
      setLivePrediction(null);
      setLiveError("");
      setIsLivePredicting(false);
      livePredictInFlightRef.current = false;
      livePredictionWindowRef.current = [];
      liveStableLabelRef.current = "";
      lastStrongSignPredictionRef.current = {
        prediction: null,
        at: 0,
      };
      lastSpokenPredictionRef.current = { key: "", spokenAt: 0 };
      lastAlphabetAppendRef.current = "";
      liveRequestRef.current += 1;
    }

    setCameraGuide((previousGuide) => (
      previousGuide.oneHandDone === nextGuide.oneHandDone &&
      previousGuide.waveDone === nextGuide.waveDone &&
      previousGuide.shoulderDone === nextGuide.shoulderDone &&
      previousGuide.headVisible === nextGuide.headVisible &&
      previousGuide.faceVisible === nextGuide.faceVisible &&
      previousGuide.eyesVisible === nextGuide.eyesVisible &&
      previousGuide.noseVisible === nextGuide.noseVisible &&
      previousGuide.lipsVisible === nextGuide.lipsVisible &&
      previousGuide.earsVisible === nextGuide.earsVisible &&
      previousGuide.neckVisible === nextGuide.neckVisible &&
      previousGuide.shoulderVisible === nextGuide.shoulderVisible &&
      previousGuide.elbowVisible === nextGuide.elbowVisible &&
      previousGuide.handVisible === nextGuide.handVisible &&
      previousGuide.fingersVisible === nextGuide.fingersVisible &&
      previousGuide.poseDetected === nextGuide.poseDetected &&
      previousGuide.handsDetectedCount === nextGuide.handsDetectedCount &&
      previousGuide.readyCount === nextGuide.readyCount
        ? previousGuide
        : nextGuide
    ));
  }

  function handleVoiceToggle() {
    if (!recognitionRef.current) {
      setVoiceStatusKey("unsupported");
      return;
    }

    if (isRecording) {
      stopVoiceRecognition(transcriptRef.current ? "finishedSaved" : "stopped");
      return;
    }

    try {
      recognitionRef.current.start();
    } catch {
      setVoiceStatusKey("startError");
    }
  }

  function renderScreen() {
    if (activeTab === "home") {
      return (
        <section className="app-screen home-view">
          <div className="home-copy">
            <span className="screen-badge">{t.home.badge}</span>
            <strong className="screen-title">{t.home.title}</strong>
            <p className="screen-text">{t.home.subtitle}</p>
            <p className="screen-subtext">{t.home.text}</p>
          </div>

          <div className="feature-grid">
            {t.home.features.map(([title, text]) => (
              <article key={title} className="feature-card">
                <strong>{title}</strong>
                <p>{text}</p>
              </article>
            ))}
          </div>
        </section>
      );
    }

    if (activeTab === "camera") {
      return (
        <section className={`app-screen camera-view ${isUserUi ? "user-camera-view" : ""}`}>
          <div className="camera-stage">
            <CameraView
              startSignal={cameraStartSignal}
              recognitionLevel={cameraRecognitionLevel}
              facingMode={cameraFacingMode}
              mirrorPreview={isCameraPreviewMirrored}
              showLandmarkOverlay={!isUserUi}
              showStatusOverlay={!isUserUi}
              onPredictionChange={handleCameraPredictionChange}
              onStreamFrame={
                cameraRecognitionLevel === "sign"
                  ? ({ ts, imageBase64 }) => {
                    sendGestureBridgeFrame({ ts, imageBase64 });
                  }
                  : undefined
              }
               onSequencePredict={async ({ frames, metadata }) => {
                const isAlphabetMode = cameraRecognitionLevel === "alphabet";
                const isPrivetMode = cameraRecognitionLevel === "sign";
                const isPrivetFocused = focusedPhraseLabel === "Привет";
                const isSignMode = cameraRecognitionLevel === "sign";

                const shouldUsePrivetDetectors =
                  cameraRecognitionLevel === "phrase" ||
                  (isPrivetMode && isPrivetFocused);
                const isWaveGreetingSequence =
                  shouldUsePrivetDetectors &&
                  isWaveGreetingMetadata(metadata);

                if (cameraRecognitionLevel === "sign") {
                  const yaPrediction = buildYaPointSelfPrediction({
                    frames,
                    basePrediction: null,
                  });

                  if (
                    yaPrediction &&
                    Number(yaPrediction.confidence ?? 0) >= 0.74
                  ) {
                    setLivePrediction({
                      ...yaPrediction,
                      sourceModel: "local-ya-point-self",
                    });
                    speakLivePrediction("Я", "sign");
                    return;
                  }

                  const spasiboPrediction = buildSpasiboThanksPrediction({
                    frames,
                    basePrediction: null,
                  });
                  const spasiboDetector = spasiboPrediction?.detector ?? {};
                  const spasiboStrong =
                    Boolean(spasiboDetector.touchDetected) &&
                    Number(spasiboDetector.fistRatio ?? 0) >= 0.18 &&
                    Number(spasiboDetector.foreheadRatio ?? 0) >= 0.14 &&
                    Number(spasiboDetector.xRangeNorm ?? 9) <= 3.2 &&
                    Number(spasiboDetector.yRangeNorm ?? 9) <= 3.2;

                  if (
                    spasiboPrediction &&
                    spasiboStrong &&
                    Number(spasiboPrediction.confidence ?? 0) >= 0.78
                  ) {
                    setLivePrediction({
                      ...spasiboPrediction,
                      sourceModel: "local-spasibo",
                    });
                    speakLivePrediction("Спасибо", "sign");
                    return;
                  }

                  const muzhchinaPrediction = buildMuzhchinaLipPinchPrediction({
                    frames,
                    basePrediction: null,
                  });

                  if (
                    muzhchinaPrediction &&
                    Number(muzhchinaPrediction.confidence ?? 0) >= 0.78
                  ) {
                    setLivePrediction({
                      ...muzhchinaPrediction,
                      sourceModel: "local-muzhchina",
                    });
                    speakLivePrediction("Мужчина", "sign");
                    return;
                  }

                  const zhenshchinaPrediction = buildZhenshchinaPalmSwipePrediction({
                    frames,
                    basePrediction: null,
                  });

                  if (
                    zhenshchinaPrediction &&
                    Number(zhenshchinaPrediction.confidence ?? 0) >= 0.7
                  ) {
                    setLivePrediction({
                      ...zhenshchinaPrediction,
                      sourceModel: "local-zhenshchina",
                    });
                    speakLivePrediction("Женщина", "sign");
                    return;
                  }

                  const pokaPrediction = buildPokaPalmPulsePrediction({
                    frames,
                    basePrediction: null,
                  });
                  const pokaDetector = pokaPrediction?.detector ?? {};
                  const pokaPulseLike =
                    Boolean(pokaDetector.pulseLike) &&
                    Number(pokaDetector.palmPulseTransitions ?? 0) >= 1 &&
                    Number(pokaDetector.closePalmRatio ?? 0) >= 0.12;
                  const pokaOpenPalmRatio = Number(pokaDetector.openPalmRatio ?? 0);

                  if (
                    pokaPrediction &&
                    pokaPulseLike &&
                    pokaOpenPalmRatio >= 0.18 &&
                    Number(pokaPrediction.confidence ?? 0) >= 0.72
                  ) {
                    setLivePrediction({
                      ...pokaPrediction,
                      sourceModel: "local-poka-pulse",
                    });
                    speakLivePrediction("Пока", "sign");
                    return;
                  }

                  if (
                    isWaveGreetingSequence ||
                    isSimpleWaveHelloMetadata(metadata)
                  ) {
                    setLivePrediction({
                      label: "Привет",
                      confidence: 0.82,
                      recognitionLevel: "sign",
                      sourceModel: "local-wave-hello",
                      scores: [{ label: "Привет", confidence: 0.82 }],
                      detector: {
                        type: "simple_wave_hello",
                        waveScore: Number(metadata?.waveScore ?? 0),
                        oneHandRatio: Number(metadata?.waveOneHandRatio ?? 0),
                        horizontalRange: Number(metadata?.waveHorizontalRange ?? 0),
                        directionChanges: Number(metadata?.waveDirectionChanges ?? 0),
                      },
                    });
                    speakLivePrediction("Привет", "sign");
                  }

                  return;
                }

                if (
                  metadata &&
                  !isWaveGreetingSequence &&
                  (Number(metadata.dominantWeightedRatio ?? 0) < (isAlphabetMode ? 0.48 : isSignMode ? 0.42 : 0.58) ||
                    Number(metadata.averageConfidence ?? 0) < (isAlphabetMode ? 0.56 : isSignMode ? 0.42 : 0.64) ||
                    Number(metadata.averageMargin ?? 0) < (isAlphabetMode ? 0.03 : isSignMode ? 0.02 : 0.05))
                ) {
                  return;
                }

                const now = Date.now();

                if (livePredictInFlightRef.current || now - liveLastRequestAtRef.current < LIVE_MIN_INTERVAL_MS) {
                  return;
                }

                liveLastRequestAtRef.current = now;
                livePredictInFlightRef.current = true;
                const requestId = liveRequestRef.current + 1;
                liveRequestRef.current = requestId;
                setIsLivePredicting(true);

                try {
                  let primaryPrediction = null;
                  const fallbackPredictions = [];
                  let lastCaughtError = null;
                  const shouldQueryPrimaryModel =
                    cameraRecognitionLevel !== "phrase" || !phraseModelUnavailableRef.current;

                  if (shouldQueryPrimaryModel) {
                    try {
                      primaryPrediction = await predictLatestModel({
                        recognitionLevel: cameraRecognitionLevel,
                        profile: "fast",
                        sequence: frames,
                        allowedRecognitionLevels: [cameraRecognitionLevel],
                      });
                    } catch (error) {
                      if (
                        cameraRecognitionLevel === "phrase" &&
                        isModelNotTrainedError(error)
                      ) {
                        phraseModelUnavailableRef.current = true;
                      }
                      lastCaughtError = error;
                    }
                  }

                  if (cameraRecognitionLevel === "phrase") {
                    for (const fallbackConfig of PHRASE_SIGN_FALLBACK_CANDIDATES) {
                      try {
                        const fallbackPrediction = await predictLatestModel({
                          recognitionLevel: "sign",
                          profile: fallbackConfig.profile,
                          sequence: frames,
                          allowedRecognitionLevels: ["sign"],
                          allowedLabelKeys: fallbackConfig.allowedLabelKeys,
                        });

                        fallbackPredictions.push({
                          prediction: fallbackPrediction,
                          config: fallbackConfig,
                        });
                      } catch (error) {
                        if (!lastCaughtError) {
                          lastCaughtError = error;
                        }
                      }
                    }
                  }

                  const prediction =
                    cameraRecognitionLevel === "phrase"
                      ? choosePhraseFallbackPrediction(
                        primaryPrediction,
                        fallbackPredictions,
                      )
                      : primaryPrediction;
                  const trajectoryPrediction =
                    shouldUsePrivetDetectors
                      ? buildPrivetTrajectoryPrediction({
                        frames,
                        metadata,
                        basePrediction: prediction,
                      })
                      : null;
                  const salutePrivetPrediction =
                    shouldUsePrivetDetectors
                      ? buildPrivetSalutePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const palmPulsePrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildPokaPalmPulsePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const pokaFastPathPrediction =
                    isPrivetMode &&
                    Boolean(palmPulsePrediction?.detector?.pulseLike) &&
                    Number(palmPulsePrediction?.detector?.palmPulseTransitions ?? 0) >= 1 &&
                    Number(palmPulsePrediction?.detector?.openPalmRatio ?? 0) >= 0.18 &&
                    Number(palmPulsePrediction?.detector?.closePalmRatio ?? 0) >= 0.12
                      ? {
                        label: "Пока",
                        confidence: Math.max(
                          0.84,
                          Number(palmPulsePrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-poka-fastpath",
                        scores: [{
                          label: "Пока",
                          confidence: Math.max(
                            0.84,
                            Number(palmPulsePrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "poka_fastpath",
                          openPalmRatio: Number(
                            palmPulsePrediction?.detector?.openPalmRatio ?? 0,
                          ),
                          closePalmRatio: Number(
                            palmPulsePrediction?.detector?.closePalmRatio ?? 0,
                          ),
                          palmPulseTransitions: Number(
                            palmPulsePrediction?.detector?.palmPulseTransitions ?? 0,
                          ),
                          pulseLike: Boolean(
                            palmPulsePrediction?.detector?.pulseLike,
                          ),
                        },
                      }
                      : null;
                  const daPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildDaNodPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const netPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildNetSnapPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const yaPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildYaPointSelfPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const tyPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildTyPointYouPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const horoshoPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildHoroshoThumbUpPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const stopPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildStopOpenPalmPrediction({
                        frames,
                        metadata,
                        basePrediction: prediction,
                      })
                      : null;
                  const estPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildEstEatPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const pitPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildPitDrinkPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const spatPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildSpatSleepPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const idtiPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildIdtiWalkPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const bezhatPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildBezhatRunPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const dumatPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildDumatThinkPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const lyubovPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildLyubovLovePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const bolshoyPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildBolshoyWidePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const malenkiyPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildMalenkiySmallPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const krasiviyPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildKrasiviyFaceGracePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const spasiboPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildSpasiboThanksPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const druzhbaPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildDruzhbaInterlockPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const druzhbaCrossPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildDruzhbaCrossedIndexPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const domPrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildDomRoofPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const solntsePrediction =
                    cameraRecognitionLevel === "phrase"
                      ? buildSolntseSunrayPrediction({
                        frames,
                        metadata,
                        basePrediction: prediction,
                      })
                      : null;
                  const zhenshchinaCheekPrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildZhenshchinaCheekPrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const zhenshchinaFastPathPrediction =
                    isPrivetMode &&
                    Number(zhenshchinaCheekPrediction?.detector?.nearCheekRatio ?? 0) >= 0.34 &&
                    Number(zhenshchinaCheekPrediction?.detector?.lowerFaceRatio ?? 0) >= 0.5 &&
                    (
                      Boolean(zhenshchinaCheekPrediction?.detector?.downwardSweep) ||
                      Boolean(zhenshchinaCheekPrediction?.detector?.sideSweep)
                    )
                      ? {
                        label: "Женщина",
                        confidence: Math.max(
                          0.84,
                          Number(zhenshchinaCheekPrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-zhenshchina-cheek-fastpath",
                        scores: [{
                          label: "Женщина",
                          confidence: Math.max(
                            0.84,
                            Number(zhenshchinaCheekPrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "zhenshchina_cheek_fastpath",
                          nearCheekRatio: Number(
                            zhenshchinaCheekPrediction?.detector?.nearCheekRatio ?? 0,
                          ),
                          lowerFaceRatio: Number(
                            zhenshchinaCheekPrediction?.detector?.lowerFaceRatio ?? 0,
                          ),
                          downwardSweep: Boolean(
                            zhenshchinaCheekPrediction?.detector?.downwardSweep,
                          ),
                          sideSweep: Boolean(
                            zhenshchinaCheekPrediction?.detector?.sideSweep,
                          ),
                        },
                      }
                      : null;
                  const bolshoyFastPathPrediction =
                    isPrivetMode &&
                    Number(bolshoyPrediction?.detector?.spreadAverage ?? 0) >= 1.38 &&
                    Number(bolshoyPrediction?.detector?.twoHandsRatio ?? 0) >= 0.28
                      ? {
                        label: "Большой",
                        confidence: Math.max(
                          0.84,
                          Number(bolshoyPrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-bolshoy-wide-fastpath",
                        scores: [{
                          label: "Большой",
                          confidence: Math.max(
                            0.84,
                            Number(bolshoyPrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "bolshoy_wide_fastpath",
                          twoHandsRatio: Number(
                            bolshoyPrediction?.detector?.twoHandsRatio ?? 0,
                          ),
                          wideRatio: Number(bolshoyPrediction?.detector?.wideRatio ?? 0),
                          spreadAverage: Number(
                            bolshoyPrediction?.detector?.spreadAverage ?? 0,
                          ),
                          openHandsRatio: Number(
                            bolshoyPrediction?.detector?.openHandsRatio ?? 0,
                          ),
                        },
                      }
                      : null;
                  const malenkiyFastPathPrediction =
                    isPrivetMode &&
                    (
                      (
                        Number(malenkiyPrediction?.detector?.gapAverage ?? 99) <= 1 &&
                        Number(malenkiyPrediction?.detector?.twoHandsRatio ?? 0) >= 0.16
                      ) ||
                      Number(malenkiyPrediction?.detector?.pinchRatio ?? 0) >= 0.24
                    )
                      ? {
                        label: "Маленький",
                        confidence: Math.max(
                          0.84,
                          Number(malenkiyPrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-malenkiy-small-fastpath",
                        scores: [{
                          label: "Маленький",
                          confidence: Math.max(
                            0.84,
                            Number(malenkiyPrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "malenkiy_small_fastpath",
                          twoHandsRatio: Number(
                            malenkiyPrediction?.detector?.twoHandsRatio ?? 0,
                          ),
                          gapAverage: Number(
                            malenkiyPrediction?.detector?.gapAverage ?? 99,
                          ),
                          pinchRatio: Number(
                            malenkiyPrediction?.detector?.pinchRatio ?? 0,
                          ),
                        },
                      }
                      : null;
                  const krasiviyFastPathPrediction =
                    isPrivetMode &&
                    Number(krasiviyPrediction?.detector?.nearFaceRatio ?? 0) >= 0.22 &&
                    (
                      Boolean(krasiviyPrediction?.detector?.downwardSweep) ||
                      Boolean(krasiviyPrediction?.detector?.softArc)
                    )
                      ? {
                        label: "Красивый",
                        confidence: Math.max(
                          0.84,
                          Number(krasiviyPrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-krasiviy-face-fastpath",
                        scores: [{
                          label: "Красивый",
                          confidence: Math.max(
                            0.84,
                            Number(krasiviyPrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "krasiviy_face_fastpath",
                          nearFaceRatio: Number(
                            krasiviyPrediction?.detector?.nearFaceRatio ?? 0,
                          ),
                          downwardSweep: Boolean(
                            krasiviyPrediction?.detector?.downwardSweep,
                          ),
                          softArc: Boolean(
                            krasiviyPrediction?.detector?.softArc,
                          ),
                        },
                      }
                      : null;
                  const spasiboFastPathPrediction =
                    isPrivetMode &&
                    (focusedPhraseLabel === "Спасибо" || focusedPhraseLabel === "Thank you") &&
                    Boolean(spasiboPrediction?.detector?.touchDetected) &&
                    Number(spasiboPrediction?.detector?.fistRatio ?? 0) >= 0.2 &&
                    Number(spasiboPrediction?.detector?.foreheadRatio ?? 0) >= 0.18 &&
                    Number(spasiboPrediction?.detector?.xRangeNorm ?? 9) <= 3.0 &&
                    Number(spasiboPrediction?.detector?.yRangeNorm ?? 9) <= 3.0
                      ? {
                        label: "Спасибо",
                        confidence: Math.max(
                          0.84,
                          Number(spasiboPrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-spasibo-fastpath",
                        scores: [{
                          label: "Спасибо",
                          confidence: Math.max(
                            0.84,
                            Number(spasiboPrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "spasibo_fastpath",
                          fistRatio: Number(
                            spasiboPrediction?.detector?.fistRatio ?? 0,
                          ),
                          foreheadRatio: Number(
                            spasiboPrediction?.detector?.foreheadRatio ?? 0,
                          ),
                          touchDetected: Boolean(
                            spasiboPrediction?.detector?.touchDetected,
                          ),
                          xRangeNorm: Number(
                            spasiboPrediction?.detector?.xRangeNorm ?? 9,
                          ),
                          yRangeNorm: Number(
                            spasiboPrediction?.detector?.yRangeNorm ?? 9,
                          ),
                        },
                      }
                      : null;
                  const moustachePrediction =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode)
                      ? buildMuzhchinaMoustachePrediction({
                        frames,
                        basePrediction: prediction,
                      })
                      : null;
                  const moustacheFastPathPrediction =
                    isPrivetMode &&
                    Number(moustachePrediction?.detector?.nearRatio ?? 0) >= 0.24 &&
                    Number(moustachePrediction?.detector?.proximityScore ?? 0) >= 0.22 &&
                    Number(moustachePrediction?.detector?.lipProximityScore ?? 0) >= 0.18
                      ? {
                        label: "Мужчина",
                        confidence: Math.max(
                          0.82,
                          Number(moustachePrediction?.confidence ?? 0),
                        ),
                        recognitionLevel: "sign",
                        sourceModel: "sign-muzhchina-moustache-fastpath",
                        scores: [{
                          label: "Мужчина",
                          confidence: Math.max(
                            0.82,
                            Number(moustachePrediction?.confidence ?? 0),
                          ),
                        }],
                        detector: {
                          type: "muzhchina_moustache_fastpath",
                          nearRatio: Number(moustachePrediction?.detector?.nearRatio ?? 0),
                          proximityScore: Number(
                            moustachePrediction?.detector?.proximityScore ?? 0,
                          ),
                          lipProximityScore: Number(
                            moustachePrediction?.detector?.lipProximityScore ?? 0,
                          ),
                          pinchRatio: Number(moustachePrediction?.detector?.pinchRatio ?? 0),
                        },
                      }
                      : null;
                  const faceAreaFallbackPrediction =
                    isPrivetMode
                      ? [
                        moustachePrediction,
                        zhenshchinaCheekPrediction,
                        krasiviyPrediction,
                        spasiboPrediction,
                      ]
                        .filter(Boolean)
                        .map((candidate) => {
                          const metrics = getPredictionMetrics(candidate);
                          const detectorScore = Number(
                            candidate?.detector?.detectorScore ?? 0,
                          );

                          return {
                            candidate,
                            score: metrics.confidence * 0.74 + detectorScore * 0.26,
                            confidence: metrics.confidence,
                            detectorScore,
                          };
                        })
                        .filter(
                          ({ confidence, detectorScore }) =>
                            confidence >= 0.7 || detectorScore >= 0.34,
                        )
                        .sort((left, right) => right.score - left.score)[0]?.candidate ?? null
                      : null;
                  const waveOnlyPrivetPrediction =
                    shouldUsePrivetDetectors &&
                    focusedPhraseLabel !== "Пока" &&
                    !trajectoryPrediction &&
                    !salutePrivetPrediction &&
                    (
                      isWaveGreetingSequence ||
                      (
                        Number(metadata?.waveScore ?? 0) >= 0.5 &&
                        Number(metadata?.waveOneHandRatio ?? 0) >= 0.54 &&
                        Number(metadata?.waveHorizontalRange ?? 0) >= 0.05 &&
                        Number(metadata?.waveHorizontalRange ?? 0) <= 0.18 &&
                        Number(metadata?.waveDirectionChanges ?? 0) <= 2
                      )
                    )
                      ? {
                        label: "Привет",
                        confidence: 0.82,
                        recognitionLevel: "sign",
                        sourceModel: "sign-privet-wave-fastpath",
                        scores: [{ label: "Привет", confidence: 0.82 }],
                        detector: {
                          type: "privet_wave_fastpath",
                          waveScore: Number(metadata?.waveScore ?? 0),
                          oneHandRatio: Number(metadata?.waveOneHandRatio ?? 0),
                          horizontalRange: Number(metadata?.waveHorizontalRange ?? 0),
                        },
                      }
                      : null;
                  let resolvedPrediction =
                    trajectoryPrediction ??
                    salutePrivetPrediction ??
                    pokaFastPathPrediction ??
                    spasiboFastPathPrediction ??
                    krasiviyFastPathPrediction ??
                    malenkiyFastPathPrediction ??
                    bolshoyFastPathPrediction ??
                    zhenshchinaFastPathPrediction ??
                    moustacheFastPathPrediction ??
                    faceAreaFallbackPrediction ??
                    waveOnlyPrivetPrediction ??
                    prediction;

                  if (salutePrivetPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const saluteMetrics = getPredictionMetrics(salutePrivetPrediction);
                    const canReplaceWithPrivetSalute =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Привет" ||
                      saluteMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        ["Женщина", "Мужчина", "Да", "Нет"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      ) ||
                      (
                        resolvedMetrics.label === "Пока" &&
                        saluteMetrics.confidence >= resolvedMetrics.confidence + 0.08 &&
                        resolvedMetrics.confidence < 0.82
                      ) ||
                      (
                        resolvedMetrics.label === "Пока" &&
                        Number(salutePrivetPrediction?.detector?.nearTempleStartRatio ?? 0) >= 0.24 &&
                        Number(salutePrivetPrediction?.detector?.outwardDelta ?? 0) >= 0.08 &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithPrivetSalute) {
                      resolvedPrediction = salutePrivetPrediction;
                    }
                  }

                  if (palmPulsePrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const palmPulseMetrics = getPredictionMetrics(palmPulsePrediction);
                    const openPalmRatio = Number(palmPulsePrediction?.detector?.openPalmRatio ?? 0);
                    const closePalmRatio = Number(palmPulsePrediction?.detector?.closePalmRatio ?? 0);
                    const palmPulseTransitions = Number(
                      palmPulsePrediction?.detector?.palmPulseTransitions ?? 0,
                    );
                    const pulseLike = Boolean(palmPulsePrediction?.detector?.pulseLike);
                    const pulseStrong =
                      pulseLike &&
                      palmPulseTransitions >= 1 &&
                      openPalmRatio >= 0.14 &&
                      closePalmRatio >= 0.12;
                    const canReplaceWithPoka =
                      pulseStrong &&
                      (
                        !resolvedMetrics.label ||
                        resolvedMetrics.label === "Пока" ||
                        (
                          resolvedMetrics.label !== "Привет" &&
                          resolvedMetrics.label !== "Мужчина" &&
                          palmPulseMetrics.confidence >= Math.max(0.72, resolvedMetrics.confidence + 0.06)
                        )
                      );

                    if (canReplaceWithPoka) {
                      resolvedPrediction = palmPulsePrediction;
                    }
                  }

                  if (stopPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const stopMetrics = getPredictionMetrics(stopPrediction);
                    const stopPoseRatio = Number(
                      stopPrediction?.detector?.stopPoseRatio ?? 0,
                    );
                    const stableStopHold = Boolean(
                      stopPrediction?.detector?.stableStopHold,
                    );
                    const canReplaceWithStop =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Стоп" ||
                      stopMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        stableStopHold &&
                        stopPoseRatio >= 0.28 &&
                        ["Пока", "Привет", "Солнце"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        stableStopHold &&
                        resolvedMetrics.confidence < 0.84
                      );

                    if (canReplaceWithStop) {
                      resolvedPrediction = stopPrediction;
                    }
                  }

                  if (daPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const daMetrics = getPredictionMetrics(daPrediction);
                    const daDetectorScore = Number(daPrediction?.detector?.detectorScore ?? 0);
                    const verticalDominance = Number(
                      daPrediction?.detector?.verticalDominance ?? 0,
                    );
                    const yRangeNorm = Number(daPrediction?.detector?.yRangeNorm ?? 0);
                    const canReplaceWithDa =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Да" ||
                      daMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        verticalDominance >= 1.1 &&
                        yRangeNorm >= 0.08 &&
                        ["Нет", "Пока", "Привет", "Солнце", "Дом", "Женщина", "Мужчина"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        daDetectorScore >= 0.74 &&
                        resolvedMetrics.confidence < 0.92
                      );

                    if (canReplaceWithDa) {
                      resolvedPrediction = daPrediction;
                    }
                  }

                  if (netPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const netMetrics = getPredictionMetrics(netPrediction);
                    const netDetectorScore = Number(netPrediction?.detector?.detectorScore ?? 0);
                    const hasSnapCycle = Boolean(netPrediction?.detector?.hasSnapCycle);
                    const compactTwoFingerRatio = Number(
                      netPrediction?.detector?.compactTwoFingerRatio ?? 0,
                    );
                    const canReplaceWithNet =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Нет" ||
                      netMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        hasSnapCycle &&
                        compactTwoFingerRatio >= 0.14 &&
                        ["Да", "Пока", "Привет", "Солнце", "Дом"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        netDetectorScore >= 0.74 &&
                        resolvedMetrics.confidence < 0.92
                      );

                    if (canReplaceWithNet) {
                      resolvedPrediction = netPrediction;
                    }
                  }

                  if (horoshoPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const horoshoMetrics = getPredictionMetrics(horoshoPrediction);
                    const thumbUpRatio = Number(
                      horoshoPrediction?.detector?.thumbUpRatio ?? 0,
                    );
                    const stableThumbHold = Boolean(
                      horoshoPrediction?.detector?.stableThumbHold,
                    );
                    const canReplaceWithHorosho =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Хорошо" ||
                      horoshoMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        stableThumbHold &&
                        thumbUpRatio >= 0.34 &&
                        ["Да", "Нет", "Ты", "Я"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        stableThumbHold &&
                        resolvedMetrics.confidence < 0.84
                      );

                    if (canReplaceWithHorosho) {
                      resolvedPrediction = horoshoPrediction;
                    }
                  }

                  if (estPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const estMetrics = getPredictionMetrics(estPrediction);
                    const mouthCycles = Number(
                      estPrediction?.detector?.mouthDirectionChanges ?? 0,
                    );
                    const canReplaceWithEst =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Есть" ||
                      estMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        mouthCycles >= 2 &&
                        ["Пить", "Думать"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithEst) {
                      resolvedPrediction = estPrediction;
                    }
                  }

                  if (pitPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const pitMetrics = getPredictionMetrics(pitPrediction);
                    const towardMouthDelta = Number(
                      pitPrediction?.detector?.towardMouthDelta ?? 0,
                    );
                    const canReplaceWithPit =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Пить" ||
                      pitMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        towardMouthDelta >= 0.06 &&
                        ["Есть", "Думать"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithPit) {
                      resolvedPrediction = pitPrediction;
                    }
                  }

                  if (spatPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const spatMetrics = getPredictionMetrics(spatPrediction);
                    const nearCheekRatio = Number(
                      spatPrediction?.detector?.nearCheekRatio ?? 0,
                    );
                    const canReplaceWithSpat =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Спать" ||
                      spatMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        nearCheekRatio >= 0.3 &&
                        ["Думать", "Женщина"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithSpat) {
                      resolvedPrediction = spatPrediction;
                    }
                  }

                  if (idtiPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const idtiMetrics = getPredictionMetrics(idtiPrediction);
                    const speedScore = Number(idtiPrediction?.detector?.speedScore ?? 0);
                    const canReplaceWithIdti =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Идти" ||
                      idtiMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        speedScore >= 0.08 &&
                        speedScore <= 0.28 &&
                        ["Бежать", "Ты"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithIdti) {
                      resolvedPrediction = idtiPrediction;
                    }
                  }

                  if (bezhatPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const bezhatMetrics = getPredictionMetrics(bezhatPrediction);
                    const speedScore = Number(bezhatPrediction?.detector?.speedScore ?? 0);
                    const canReplaceWithBezhat =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Бежать" ||
                      bezhatMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        speedScore >= 0.2 &&
                        ["Идти", "Ты"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithBezhat) {
                      resolvedPrediction = bezhatPrediction;
                    }
                  }

                  if (dumatPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const dumatMetrics = getPredictionMetrics(dumatPrediction);
                    const nearTempleRatio = Number(
                      dumatPrediction?.detector?.nearTempleRatio ?? 0,
                    );
                    const canReplaceWithDumat =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Думать" ||
                      dumatMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        nearTempleRatio >= 0.28 &&
                        ["Мужчина"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithDumat) {
                      resolvedPrediction = dumatPrediction;
                    }
                  }

                  if (lyubovPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const lyubovMetrics = getPredictionMetrics(lyubovPrediction);
                    const crossRatio = Number(
                      lyubovPrediction?.detector?.crossRatio ?? 0,
                    );
                    const symmetricRatio = Number(
                      lyubovPrediction?.detector?.symmetricRatio ?? 0,
                    );
                    const chestLevelRatio = Number(
                      lyubovPrediction?.detector?.chestLevelRatio ?? 0,
                    );
                    const holdRatio = Number(
                      lyubovPrediction?.detector?.holdRatio ?? 0,
                    );
                    const canReplaceWithLyubov =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Любовь" ||
                      lyubovMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        crossRatio >= 0.34 &&
                        symmetricRatio >= 0.24 &&
                        chestLevelRatio >= 0.28 &&
                        holdRatio >= 0.48 &&
                        ["Дружба", "Дом"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.92
                      );

                    if (canReplaceWithLyubov) {
                      resolvedPrediction = lyubovPrediction;
                    }
                  }

                  if (druzhbaPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const druzhbaMetrics = getPredictionMetrics(druzhbaPrediction);
                    const twoHandsSignal =
                      Number(druzhbaPrediction?.detector?.twoHandsRatio ?? 0) >= 0.58;
                    const druzhbaDetectorScore = Number(
                      druzhbaPrediction?.detector?.detectorScore ?? 0,
                    );
                    const canReplaceWithDruzhba =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Дружба" ||
                      druzhbaMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (twoHandsSignal && resolvedMetrics.confidence < 0.84) ||
                      (
                        druzhbaDetectorScore >= 0.72 &&
                        ["Привет", "Пока", "Мужчина", "Женщина", "Дом", "Солнце"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.93
                      ) ||
                      (
                        ["Привет", "Пока", "Мужчина", "Женщина"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      );

                    if (canReplaceWithDruzhba) {
                      resolvedPrediction = druzhbaPrediction;
                    }
                  }

                  if (druzhbaCrossPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const druzhbaCrossMetrics = getPredictionMetrics(druzhbaCrossPrediction);
                    const crossRatio = Number(
                      druzhbaCrossPrediction?.detector?.crossedRatio ?? 0,
                    );
                    const canReplaceWithDruzhbaCross =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Дружба" ||
                      druzhbaCrossMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        crossRatio >= 0.24 &&
                        ["Дом", "Женщина", "Мужчина", "Привет", "Пока"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithDruzhbaCross) {
                      resolvedPrediction = druzhbaCrossPrediction;
                    }
                  }

                  if (domPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const domMetrics = getPredictionMetrics(domPrediction);
                    const domRoofSignal =
                      Number(domPrediction?.detector?.roofShapeRatio ?? 0) >= 0.52;
                    const domDetectorScore = Number(
                      domPrediction?.detector?.detectorScore ?? 0,
                    );
                    const canReplaceWithDom =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Дом" ||
                      domMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        domRoofSignal &&
                        resolvedMetrics.confidence < 0.92 &&
                        ["Дружба", "Женщина", "Мужчина", "Привет", "Пока", "Солнце"].includes(
                          resolvedMetrics.label,
                        )
                      ) ||
                      (
                        domDetectorScore >= 0.72 &&
                        resolvedMetrics.label === "Дружба" &&
                        resolvedMetrics.confidence < 0.95
                      );

                    if (canReplaceWithDom) {
                      resolvedPrediction = domPrediction;
                    }
                  }

                  if (solntsePrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const solntseMetrics = getPredictionMetrics(solntsePrediction);
                    const aboveHeadSignal =
                      Number(solntsePrediction?.detector?.aboveHeadRatio ?? 0) >= 0.56;
                    const waveScore = Number(solntsePrediction?.detector?.waveScore ?? 0);
                    const canReplaceWithSolntse =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Солнце" ||
                      solntseMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (aboveHeadSignal && resolvedMetrics.confidence < 0.9 && waveScore < 0.74) ||
                      (
                        ["Привет", "Пока", "Женщина", "Дом"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithSolntse) {
                      resolvedPrediction = solntsePrediction;
                    }
                  }

                  if (bolshoyPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const bolshoyMetrics = getPredictionMetrics(bolshoyPrediction);
                    const spreadAverage = Number(
                      bolshoyPrediction?.detector?.spreadAverage ?? 0,
                    );
                    const openHandsRatio = Number(
                      bolshoyPrediction?.detector?.openHandsRatio ?? 0,
                    );
                    const twoHandsRatio = Number(
                      bolshoyPrediction?.detector?.twoHandsRatio ?? 0,
                    );
                    const canReplaceWithBolshoy =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Большой" ||
                      bolshoyMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        spreadAverage >= 1.38 &&
                        twoHandsRatio >= 0.28 &&
                        ["Дом", "Дружба", "Любовь", "Женщина", "Мужчина"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        spreadAverage >= 1.56 &&
                        twoHandsRatio >= 0.22 &&
                        openHandsRatio >= 0.16 &&
                        resolvedMetrics.confidence < 0.92
                      );

                    if (canReplaceWithBolshoy) {
                      resolvedPrediction = bolshoyPrediction;
                    }
                  }

                  if (malenkiyPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const malenkiyMetrics = getPredictionMetrics(malenkiyPrediction);
                    const gapAverage = Number(
                      malenkiyPrediction?.detector?.gapAverage ?? 99,
                    );
                    const pinchRatio = Number(
                      malenkiyPrediction?.detector?.pinchRatio ?? 0,
                    );
                    const twoHandsRatio = Number(
                      malenkiyPrediction?.detector?.twoHandsRatio ?? 0,
                    );
                    const canReplaceWithMalenkiy =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Маленький" ||
                      malenkiyMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        gapAverage <= 1 &&
                        twoHandsRatio >= 0.16 &&
                        ["Большой", "Дом", "Дружба", "Любовь"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        pinchRatio >= 0.24 &&
                        resolvedMetrics.confidence < 0.9
                      );

                    if (canReplaceWithMalenkiy) {
                      resolvedPrediction = malenkiyPrediction;
                    }
                  }

                  if (krasiviyPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const krasiviyMetrics = getPredictionMetrics(krasiviyPrediction);
                    const nearFaceRatio = Number(
                      krasiviyPrediction?.detector?.nearFaceRatio ?? 0,
                    );
                    const yRangeNorm = Number(
                      krasiviyPrediction?.detector?.yRangeNorm ?? 0,
                    );
                    const canReplaceWithKrasiviy =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Красивый" ||
                      krasiviyMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        nearFaceRatio >= 0.22 &&
                        yRangeNorm <= 0.94 &&
                        ["Женщина", "Мужчина", "Привет", "Стоп", "Спасибо"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.9
                      ) ||
                      (
                        resolvedMetrics.label === "Спасибо" &&
                        nearFaceRatio >= 0.2 &&
                        yRangeNorm <= 0.9 &&
                        resolvedMetrics.confidence < 0.88
                      );

                    if (canReplaceWithKrasiviy) {
                      resolvedPrediction = krasiviyPrediction;
                    }
                  }

                  if (spasiboPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const spasiboMetrics = getPredictionMetrics(spasiboPrediction);
                    const fistRatio = Number(
                      spasiboPrediction?.detector?.fistRatio ?? 0,
                    );
                    const foreheadRatio = Number(
                      spasiboPrediction?.detector?.foreheadRatio ?? 0,
                    );
                    const touchDetected = Boolean(
                      spasiboPrediction?.detector?.touchDetected,
                    );
                    const xRangeNorm = Number(
                      spasiboPrediction?.detector?.xRangeNorm ?? 9,
                    );
                    const yRangeNorm = Number(
                      spasiboPrediction?.detector?.yRangeNorm ?? 9,
                    );
                    const canReplaceWithSpasibo =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Спасибо" ||
                      spasiboMetrics.confidence >= resolvedMetrics.confidence + 0.08 ||
                      (
                        touchDetected &&
                        fistRatio >= 0.24 &&
                        foreheadRatio >= 0.18 &&
                        xRangeNorm <= 3.0 &&
                        yRangeNorm <= 3.0 &&
                        ["Есть", "Пить"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.82
                      ) ||
                      (
                        resolvedMetrics.label === "Красивый" &&
                        touchDetected &&
                        foreheadRatio >= 0.2 &&
                        xRangeNorm <= 3.0 &&
                        yRangeNorm <= 3.0 &&
                        resolvedMetrics.confidence < 0.8
                      );

                    if (canReplaceWithSpasibo) {
                      resolvedPrediction = spasiboPrediction;
                    }
                  }

                  if (zhenshchinaCheekPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const zhenshchinaMetrics = getPredictionMetrics(zhenshchinaCheekPrediction);
                    const sideSweep = Boolean(zhenshchinaCheekPrediction?.detector?.sideSweep);
                    const downwardSweep = Boolean(
                      zhenshchinaCheekPrediction?.detector?.downwardSweep,
                    );
                    const xDirectionChanges = Number(
                      zhenshchinaCheekPrediction?.detector?.xDirectionChanges ?? 0,
                    );
                    const moustacheNearRatio = Number(
                      moustachePrediction?.detector?.nearRatio ?? 0,
                    );
                    const moustacheLipProximityScore = Number(
                      moustachePrediction?.detector?.lipProximityScore ?? 0,
                    );
                    const moustachePinchRatio = Number(
                      moustachePrediction?.detector?.pinchRatio ?? 0,
                    );
                    const strongMoustacheConflict =
                      moustachePrediction &&
                      moustacheNearRatio >= 0.36 &&
                      moustacheLipProximityScore >= 0.28 &&
                      moustachePinchRatio >= 0.08;
                    const canReplaceWithZhenshchina =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Женщина" ||
                      zhenshchinaMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        (sideSweep || downwardSweep) &&
                        xDirectionChanges >= 0 &&
                        ["Мужчина", "Привет", "Пока"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      );

                    if (
                      canReplaceWithZhenshchina &&
                      (
                        !strongMoustacheConflict ||
                        zhenshchinaMetrics.confidence >=
                          getPredictionMetrics(moustachePrediction).confidence + 0.06
                      )
                    ) {
                      resolvedPrediction = zhenshchinaCheekPrediction;
                    }
                  }

                  if (moustachePrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const moustacheMetrics = getPredictionMetrics(moustachePrediction);
                    const nearRatio = Number(
                      moustachePrediction?.detector?.nearRatio ?? 0,
                    );
                    const proximityScore = Number(
                      moustachePrediction?.detector?.proximityScore ?? 0,
                    );
                    const lipProximityScore = Number(
                      moustachePrediction?.detector?.lipProximityScore ?? 0,
                    );
                    const zhenshchinaNearCheekRatio = Number(
                      zhenshchinaCheekPrediction?.detector?.nearCheekRatio ?? 0,
                    );
                    const zhenshchinaLowerFaceRatio = Number(
                      zhenshchinaCheekPrediction?.detector?.lowerFaceRatio ?? 0,
                    );
                    const zhenshchinaDownwardSweep = Boolean(
                      zhenshchinaCheekPrediction?.detector?.downwardSweep,
                    );
                    const strongCheekConflict =
                      zhenshchinaCheekPrediction &&
                      zhenshchinaNearCheekRatio >= 0.36 &&
                      zhenshchinaLowerFaceRatio >= 0.52 &&
                      zhenshchinaDownwardSweep;
                    const canReplaceWithMuzhchina =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Мужчина" ||
                      moustacheMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        nearRatio >= 0.34 &&
                        proximityScore >= 0.3 &&
                        lipProximityScore >= 0.24 &&
                        ["Привет", "Женщина", "Нет", "Стоп"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      ) ||
                      (resolvedMetrics.label !== "Привет" && resolvedMetrics.confidence < 0.78) ||
                      (resolvedMetrics.label === "Привет" &&
                        resolvedMetrics.confidence < 0.72 &&
                        moustacheMetrics.confidence >= resolvedMetrics.confidence + 0.08);

                    if (
                      canReplaceWithMuzhchina &&
                      (
                        !strongCheekConflict ||
                        moustacheMetrics.confidence >=
                          getPredictionMetrics(zhenshchinaCheekPrediction).confidence + 0.06
                      )
                    ) {
                      resolvedPrediction = moustachePrediction;
                    }
                  }

                  if (yaPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const yaMetrics = getPredictionMetrics(yaPrediction);
                    const nearChestRatio = Number(
                      yaPrediction?.detector?.nearChestRatio ?? 0,
                    );
                    const chestTouchRatio = Number(
                      yaPrediction?.detector?.chestTouchRatio ?? 0,
                    );
                    const inwardPointRatio = Number(
                      yaPrediction?.detector?.inwardPointRatio ?? 0,
                    );
                    const centerOffsetAverage = Number(
                      yaPrediction?.detector?.centerOffsetAverage ?? 1,
                    );
                    const tyOutwardRatio = Number(
                      tyPrediction?.detector?.outwardRatio ?? 0,
                    );
                    const tySelfLikeRatio = Number(
                      tyPrediction?.detector?.selfLikeRatio ?? 1,
                    );
                    const tyOutwardMarginAverage = Number(
                      tyPrediction?.detector?.outwardMarginAverage ?? 0,
                    );
                    const strongTyConflict =
                      tyPrediction &&
                      tyOutwardRatio >= 0.28 &&
                      tySelfLikeRatio <= 0.38 &&
                      tyOutwardMarginAverage >= 0.12;
                    const canReplaceWithYa =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Я" ||
                      yaMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        isPrivetMode &&
                        nearChestRatio >= 0.18 &&
                        chestTouchRatio >= 0.1 &&
                        inwardPointRatio >= 0.42 &&
                        centerOffsetAverage <= 0.64 &&
                        resolvedMetrics.confidence < 0.94
                      ) ||
                      (
                        nearChestRatio >= 0.16 &&
                        chestTouchRatio >= 0.1 &&
                        inwardPointRatio >= 0.4 &&
                        centerOffsetAverage <= 0.68 &&
                        ["Ты", "Привет", "Стоп", "Нет"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      );

                    if (
                      canReplaceWithYa &&
                      (
                        !strongTyConflict ||
                        yaMetrics.confidence >= getPredictionMetrics(tyPrediction).confidence + 0.05
                      )
                    ) {
                      resolvedPrediction = yaPrediction;
                    }
                  }

                  if (tyPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const tyMetrics = getPredictionMetrics(tyPrediction);
                    const outwardRatio = Number(
                      tyPrediction?.detector?.outwardRatio ?? 0,
                    );
                    const selfLikeRatio = Number(
                      tyPrediction?.detector?.selfLikeRatio ?? 0,
                    );
                    const outwardMarginAverage = Number(
                      tyPrediction?.detector?.outwardMarginAverage ?? 0,
                    );
                    const yaNearChestRatio = Number(
                      yaPrediction?.detector?.nearChestRatio ?? 0,
                    );
                    const yaChestTouchRatio = Number(
                      yaPrediction?.detector?.chestTouchRatio ?? 0,
                    );
                    const yaInwardPointRatio = Number(
                      yaPrediction?.detector?.inwardPointRatio ?? 0,
                    );
                    const yaCenterOffsetAverage = Number(
                      yaPrediction?.detector?.centerOffsetAverage ?? 1,
                    );
                    const strongYaConflict =
                      yaPrediction &&
                      yaNearChestRatio >= 0.2 &&
                      yaChestTouchRatio >= 0.1 &&
                      yaInwardPointRatio >= 0.46 &&
                      yaCenterOffsetAverage <= 0.64;
                    const canReplaceWithTy =
                      !resolvedMetrics.label ||
                      resolvedMetrics.label === "Ты" ||
                      tyMetrics.confidence >= resolvedMetrics.confidence + 0.04 ||
                      (
                        outwardRatio >= 0.26 &&
                        selfLikeRatio <= 0.42 &&
                        outwardMarginAverage >= 0.12 &&
                        ["Я", "Привет", "Стоп"].includes(
                          resolvedMetrics.label,
                        ) &&
                        resolvedMetrics.confidence < 0.88
                      );

                    if (
                      canReplaceWithTy &&
                      (
                        !strongYaConflict ||
                        tyMetrics.confidence >= getPredictionMetrics(yaPrediction).confidence + 0.05
                      )
                    ) {
                      resolvedPrediction = tyPrediction;
                    }
                  }

                  if (focusedPhraseLabel) {
                    resolvedPrediction = chooseFocusedPhrasePrediction({
                      focusLabel: focusedPhraseLabel,
                      resolvedPrediction,
                      candidates: [
                        resolvedPrediction,
                        prediction,
                        trajectoryPrediction,
                        salutePrivetPrediction,
                        palmPulsePrediction,
                        stopPrediction,
                        estPrediction,
                        pitPrediction,
                        spatPrediction,
                        idtiPrediction,
                        bezhatPrediction,
                        dumatPrediction,
                        lyubovPrediction,
                        bolshoyPrediction,
                        bolshoyFastPathPrediction,
                        malenkiyPrediction,
                        malenkiyFastPathPrediction,
                        krasiviyPrediction,
                        krasiviyFastPathPrediction,
                        spasiboPrediction,
                        spasiboFastPathPrediction,
                        daPrediction,
                        netPrediction,
                        horoshoPrediction,
                        druzhbaPrediction,
                        druzhbaCrossPrediction,
                        domPrediction,
                        solntsePrediction,
                        zhenshchinaCheekPrediction,
                        moustachePrediction,
                        yaPrediction,
                        tyPrediction,
                      ].filter(Boolean),
                    });
                  }

                  if (cameraRecognitionLevel === "sign") {
                    resolvedPrediction = chooseBestActiveSignPrediction({
                      resolvedPrediction,
                      focusLabel: focusedPhraseLabel,
                      candidates: [
                        resolvedPrediction,
                        prediction,
                        trajectoryPrediction,
                        salutePrivetPrediction,
                        palmPulsePrediction,
                        pokaFastPathPrediction,
                        faceAreaFallbackPrediction,
                        krasiviyPrediction,
                        krasiviyFastPathPrediction,
                        spasiboPrediction,
                        spasiboFastPathPrediction,
                        zhenshchinaCheekPrediction,
                        zhenshchinaFastPathPrediction,
                        moustachePrediction,
                        moustacheFastPathPrediction,
                        yaPrediction,
                        tyPrediction,
                        bolshoyPrediction,
                        bolshoyFastPathPrediction,
                        malenkiyPrediction,
                        malenkiyFastPathPrediction,
                        waveOnlyPrivetPrediction,
                      ].filter(Boolean),
                    });

                    const resolvedLabel = String(
                      getPredictionMetrics(resolvedPrediction).label ?? "",
                    ).trim();

                    if (
                      resolvedPrediction &&
                      shouldRejectActiveSignPrediction({
                        prediction: resolvedPrediction,
                        focusLabel: focusedPhraseLabel,
                      })
                    ) {
                      resolvedPrediction = null;
                    }

                    if (resolvedLabel && !ACTIVE_SIGN_LABELS.has(resolvedLabel)) {
                      resolvedPrediction = null;
                    }
                  }

                  if (cameraRecognitionLevel === "sign" && resolvedPrediction) {
                    const resolvedMetrics = getPredictionMetrics(resolvedPrediction);
                    const affinityScore = getSignDetectorAffinity(resolvedPrediction);

                    if (
                      ACTIVE_SIGN_LABELS.has(String(resolvedMetrics.label ?? "").trim()) &&
                      (
                        resolvedMetrics.confidence >= 0.74 ||
                        affinityScore >= 0.56
                      )
                    ) {
                      lastStrongSignPredictionRef.current = {
                        prediction: resolvedPrediction,
                        at: Date.now(),
                      };
                    }
                  }

                  if (!resolvedPrediction) {
                    if (cameraRecognitionLevel === "sign") {
                      const stickyPrediction = lastStrongSignPredictionRef.current?.prediction ?? null;
                      const stickyAt = Number(lastStrongSignPredictionRef.current?.at ?? 0);
                      const stickyAge = Date.now() - stickyAt;
                      const stickyLabel = String(
                        getPredictionMetrics(stickyPrediction).label ?? "",
                      ).trim();

                      if (
                        stickyPrediction &&
                        stickyAge <= SIGN_PREDICTION_STICKY_MS &&
                        ACTIVE_SIGN_LABELS.has(stickyLabel)
                      ) {
                        resolvedPrediction = stickyPrediction;
                      }
                    }
                  }

                  if (!resolvedPrediction) {
                    if (
                      cameraRecognitionLevel === "phrase" &&
                      phraseModelUnavailableRef.current
                    ) {
                      livePredictionWindowRef.current = [];
                      liveStableLabelRef.current = "";
                      setLivePrediction(null);
                      setLiveError("");
                      return;
                    }
                    throw lastCaughtError || new Error("Не удалось получить результат распознавания");
                  }

                  if (liveRequestRef.current !== requestId) {
                    return;
                  }

                  const { confidence, margin, label } = getPredictionMetrics(resolvedPrediction);
                  const isAllowedLabel =
                    cameraRecognitionLevel === "alphabet" ||
                    ALLOWED_WORD_LABELS.has(String(label ?? "").trim());

                  if (!isAllowedLabel) {
                    livePredictionWindowRef.current = [];
                    liveStableLabelRef.current = "";
                    setLivePrediction(null);
                    setLiveError("");
                    return;
                  }

                  if (label) {
                    livePredictionWindowRef.current = [
                      ...livePredictionWindowRef.current,
                      { label, confidence, margin },
                    ].slice(-LIVE_WINDOW);
                  }

                  const summary = summarizeLiveWindow(livePredictionWindowRef.current);
                  const livePromoteThresholds = getLivePromoteThresholds({
                    recognitionLevel: cameraRecognitionLevel,
                    label: summary.label,
                  });
                  const isFocusedSummaryLabel =
                    cameraRecognitionLevel === "sign" &&
                    summary.label &&
                    String(summary.label).trim() === String(focusedPhraseLabel ?? "").trim();
                  const canPromoteInstantly = canInstantPromotePrediction({
                    recognitionLevel: cameraRecognitionLevel,
                    prediction: resolvedPrediction,
                    liveEntries: livePredictionWindowRef.current,
                    summary,
                    isWaveGreetingSequence,
                    focusLabel: focusedPhraseLabel,
                  });
                  const canPromote =
                    livePredictionWindowRef.current.length >= livePromoteThresholds.minEntries &&
                    summary.label &&
                    summary.ratio >= (
                      isFocusedSummaryLabel
                        ? Math.max(0.42, livePromoteThresholds.minRatio - 0.04)
                        : livePromoteThresholds.minRatio
                    ) &&
                    summary.weightedRatio >= (
                      isFocusedSummaryLabel
                        ? Math.max(0.42, livePromoteThresholds.minWeightedRatio - 0.04)
                        : livePromoteThresholds.minWeightedRatio
                    ) &&
                    summary.averageConfidence >= (
                      isFocusedSummaryLabel
                        ? Math.max(0.34, livePromoteThresholds.minConfidence - 0.04)
                        : livePromoteThresholds.minConfidence
                    ) &&
                    summary.averageMargin >= (
                      isFocusedSummaryLabel
                        ? Math.max(0.015, livePromoteThresholds.minMargin - 0.01)
                        : livePromoteThresholds.minMargin
                    );
                  const canPromoteWaveGreeting =
                    (cameraRecognitionLevel === "phrase" || isPrivetMode) &&
                    isWaveGreetingSequence &&
                    summary.label === "Привет" &&
                    livePredictionWindowRef.current.length >= 2 &&
                    summary.weightedRatio >= 0.5 &&
                    summary.averageConfidence >= 0.46;

                  if (canPromoteInstantly || canPromote || canPromoteWaveGreeting) {
                    liveStableLabelRef.current = summary.label;
                    if (cameraRecognitionLevel === "alphabet") {
                      appendAlphabetToWord(summary.label);
                    }
                    speakLivePrediction(summary.label, cameraRecognitionLevel);
                    setLivePrediction({
                      ...resolvedPrediction,
                      label: summary.label,
                      confidence: Math.max(confidence, summary.averageConfidence),
                    });
                  } else if (!liveStableLabelRef.current) {
                    setLivePrediction(resolvedPrediction);
                  } else if (
                    cameraRecognitionLevel === "sign" &&
                    lastStrongSignPredictionRef.current?.prediction
                  ) {
                    const stickyPrediction = lastStrongSignPredictionRef.current.prediction;
                    const stickyAge = Date.now() - Number(lastStrongSignPredictionRef.current.at ?? 0);

                    if (stickyAge <= SIGN_PREDICTION_STICKY_MS) {
                      setLivePrediction(stickyPrediction);
                    }
                  }

                  setLiveError("");
                } catch (error) {
                  if (liveRequestRef.current === requestId) {
                    setLiveError(error?.message || "Не удалось получить результат распознавания");
                  }
                } finally {
                  livePredictInFlightRef.current = false;
                  if (liveRequestRef.current === requestId) {
                    setIsLivePredicting(false);
                  }
                }
              }}
            />

            <div className="camera-overlay-dock">
              {isUserUi ? (
                <div className="camera-user-prediction-stack">
                  {renderCameraPredictionPanel()}
                </div>
              ) : null}
              <div className="camera-control-row">
                <button
                  className="camera-toggle-button"
                  type="button"
                  onClick={() => handleCameraRecognitionChange("alphabet")}
                  aria-pressed={cameraRecognitionLevel === "alphabet"}
                >
                  {t.camera.alphabet}
                </button>
                <button
                  className="camera-toggle-button"
                  type="button"
                  onClick={() => handleCameraRecognitionChange("sign")}
                  aria-pressed={cameraRecognitionLevel === "sign"}
                >
                  {t.camera.sign}
                </button>
              </div>

              <div className="camera-action-row">
                <button className="primary-screen-button" type="button" onClick={handleCameraToggle}>
                  {cameraStartSignal > 0 ? t.camera.stop : t.camera.start}
                </button>
                <button
                  className="camera-toggle-button camera-icon-button"
                  type="button"
                  onClick={handleCameraFacingToggle}
                  aria-label={cameraFacingMode === "user" ? t.camera.rear : t.camera.front}
                  title={cameraFacingMode === "user" ? t.camera.rear : t.camera.front}
                >
                  <CameraSwitchIcon />
                </button>
              </div>
            </div>
          </div>
        </section>
      );
    }

    if (activeTab === "voice") {
      return (
        <section className="app-screen voice-view">
          <div className="voice-header">
            <span className="screen-badge">{t.voice.badge}</span>
            <strong className="screen-title">{t.voice.title}</strong>
            <p className="screen-text">{t.voice.text}</p>
          </div>

          <section className={`voice-recorder-panel ${isRecording ? "recording" : ""}`}>
            <button
              className={`voice-mic-button ${isRecording ? "recording" : ""}`}
              type="button"
              onClick={handleVoiceToggle}
              aria-label={t.voice.micLabel}
              aria-pressed={isRecording}
            >
              <VoiceIcon />
            </button>
            <p className="voice-recording-label">
              {isRecording ? t.voice.recording : t.voice.start}
            </p>
            <p className="voice-status">{voiceStatusText}</p>
          </section>

          <article className="voice-result-card">
            <strong className="voice-result-title">{t.voice.transcript}</strong>
            <div className="voice-result-main">
              {transcript || interimTranscript ? (
                <>
                  {transcript ? <span>{transcript}</span> : null}
                  {interimTranscript ? <span className="voice-result-interim"> {interimTranscript}</span> : null}
                </>
              ) : (
                <span className="voice-result-empty">{t.voice.transcriptEmpty}</span>
              )}
            </div>
            <p className="voice-result-hint">{t.voice.transcriptHint}</p>
          </article>

          <button className="primary-screen-button" type="button" onClick={handleVoiceToggle}>
            {isRecording ? t.voice.stop : t.voice.start}
          </button>
        </section>
      );
    }

    return (
      <section className="app-screen settings-view">
        <div className="placeholder-header">
          <span className="screen-badge">{t.settings.badge}</span>
          <strong className="screen-title">{t.settings.title}</strong>
          <p className="screen-text">{t.settings.text}</p>
        </div>

        <article className="settings-card">
          <div className="settings-card-head">
            <div>
              <strong className="settings-card-title">{t.settings.themeTitle}</strong>
              <p className="settings-card-text">{t.settings.themeText}</p>
            </div>
            <span className="settings-current-pill">{t.options.theme[theme]}</span>
          </div>
          <div className="settings-segmented">
            {["dark", "light"].map((option) => (
              <button
                key={option}
                className={`segment-button ${theme === option ? "active" : ""}`}
                type="button"
                onClick={() => setTheme(option)}
              >
                {t.options.theme[option]}
              </button>
            ))}
          </div>
        </article>

        <article className="settings-card">
          <div className="settings-card-head">
            <div>
              <strong className="settings-card-title">{t.settings.uiModeTitle}</strong>
              <p className="settings-card-text">{t.settings.uiModeText}</p>
            </div>
            <span className="settings-current-pill">
              {isUserUi ? t.options.uiMode.user : t.options.uiMode.admin}
            </span>
          </div>
          <div className="settings-segmented">
            {(["admin", "user"]).map((option) => (
              <button
                key={option}
                className={`segment-button ${
                  (option === "user") === isUserUi ? "active" : ""
                }`}
                type="button"
                onClick={() => switchUiMode(option)}
              >
                {t.options.uiMode[option]}
              </button>
            ))}
          </div>
        </article>

        <article className="settings-card">
          <div className="settings-card-head">
            <div>
              <strong className="settings-card-title">{t.settings.autoSpeakTitle}</strong>
              <p className="settings-card-text">{t.settings.autoSpeakText}</p>
            </div>
            <span className="settings-current-pill">
              {autoSpeakEnabled ? t.options.autoSpeak.on : t.options.autoSpeak.off}
            </span>
          </div>
          <div className="settings-segmented">
            {[true, false].map((enabled) => (
              <button
                key={enabled ? "on" : "off"}
                className={`segment-button ${autoSpeakEnabled === enabled ? "active" : ""}`}
                type="button"
                onClick={() => setAutoSpeakEnabled(enabled)}
              >
                {enabled ? t.options.autoSpeak.on : t.options.autoSpeak.off}
              </button>
            ))}
          </div>
        </article>

        <article className="settings-card">
          <div className="settings-card-head">
            <div>
              <strong className="settings-card-title">{t.settings.languageTitle}</strong>
              <p className="settings-card-text">{t.settings.languageText}</p>
            </div>
            <span className="settings-current-pill">{t.options.language[locale]}</span>
          </div>
          <div className="settings-segmented">
            {["ru", "en"].map((option) => (
              <button
                key={option}
                className={`segment-button ${locale === option ? "active" : ""}`}
                type="button"
                onClick={() => {
                  setLocale(option);
                  setVoiceStatusKey(getSpeechRecognitionApi() ? "languageChanged" : "unsupported");
                }}
              >
                {t.options.language[option]}
              </button>
            ))}
          </div>
        </article>

        <article className="settings-card settings-summary-card">
          <span className="transcript-kicker">{t.settings.previewTitle}</span>
          <div className="settings-summary-grid">
            <div className="settings-summary-row">
              <span className="settings-summary-label">Тема</span>
              <strong className="settings-summary-value">{t.options.theme[theme]}</strong>
            </div>
            <div className="settings-summary-row">
              <span className="settings-summary-label">{t.settings.uiModeTitle}</span>
              <strong className="settings-summary-value">
                {isUserUi ? t.options.uiMode.user : t.options.uiMode.admin}
              </strong>
            </div>
            <div className="settings-summary-row">
              <span className="settings-summary-label">{t.settings.autoSpeakTitle}</span>
              <strong className="settings-summary-value">
                {autoSpeakEnabled ? t.options.autoSpeak.on : t.options.autoSpeak.off}
              </strong>
            </div>
            <div className="settings-summary-row">
              <span className="settings-summary-label">Язык</span>
              <strong className="settings-summary-value">{t.options.language[locale]}</strong>
            </div>
          </div>
          <p className="settings-card-text">{t.settings.previewText}</p>
        </article>
      </section>
    );
  }

  if (useNativeMobileShell) {
    return (
      <main className={`native-mobile-stage theme-${theme}`}>
        <section className="native-mobile-screen">
          <div className="screen-canvas native-mobile-shell">
            {renderScreen()}
            <nav className="bottom-nav" aria-label={locale === "ru" ? "Нижняя навигация" : "Bottom navigation"}>
              {tabs.map((item) => {
                const IconComponent = item.Icon;

                return (
                  <button
                    key={item.id}
                    className={`nav-item ${activeTab === item.id ? "active" : ""}`}
                    type="button"
                    onClick={() => handleTabChange(item.id)}
                  >
                    <span className="nav-icon" aria-hidden="true">
                      <IconComponent />
                    </span>
                    <span className="nav-label">{t.nav[item.id]}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main
      className={`device-stage theme-${theme} ${showExternalCameraPhraseLibrary ? "with-camera-guide" : ""}`}
    >
      {!isUserUi ? (
        <div className="ui-mode-shell">
          {renderUiModeSwitch(isUserUi)}
        </div>
      ) : null}
      <section className="phone-mockup" aria-label="Макет телефона">
        <div className="phone-frame">
          <div className="phone-screen">
            <div className="status-bar">
              <span className="status-time">9:41</span>
              <div className="status-icons">
                <span className="signal-bars" />
                <span className="wifi-icon" />
                <span className="battery-icon"><span className="battery-level" /></span>
              </div>
            </div>

            <div className="dynamic-island" aria-hidden="true" />

            <div className="screen-canvas">
              {renderScreen()}
              <nav className="bottom-nav" aria-label={locale === "ru" ? "Нижняя навигация" : "Bottom navigation"}>
                {tabs.map((item) => {
                  const IconComponent = item.Icon;

                  return (
                    <button
                      key={item.id}
                      className={`nav-item ${activeTab === item.id ? "active" : ""}`}
                      type="button"
                      onClick={() => handleTabChange(item.id)}
                    >
                      <span className="nav-icon" aria-hidden="true">
                        <IconComponent />
                      </span>
                      <span className="nav-label">{t.nav[item.id]}</span>
                    </button>
                  );
                })}
              </nav>
            </div>

            <div className="home-indicator" aria-hidden="true" />
          </div>
        </div>
      </section>

      {showExternalCameraPhraseLibrary ? (
        <aside className="camera-phrases-side" aria-label={locale === "ru" ? "Список слов" : "Word list"}>
          {renderCameraPhraseLibraryPanel()}
        </aside>
      ) : null}

      {showExternalCameraVisionPanel ? (
        <aside className="camera-guide-side" aria-label={locale === "ru" ? "Точки, которые видит ИИ" : "AI visible points"}>
          {renderCameraVisionPanel()}
        </aside>
      ) : null}
    </main>
  );
}

function HomeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M4.5 10.5 12 4l7.5 6.5v8A1.5 1.5 0 0 1 18 20h-3.5v-5h-5v5H6A1.5 1.5 0 0 1 4.5 18.5v-8Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M7 8.5h2l1.2-1.5h3.6L15 8.5h2A2.5 2.5 0 0 1 19.5 11v6A2.5 2.5 0 0 1 17 19.5H7A2.5 2.5 0 0 1 4.5 17v-6A2.5 2.5 0 0 1 7 8.5Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <circle cx="12" cy="14" r="2.8" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function CameraSwitchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none">
      <path
        d="M7 8.5h3l1.3-1.6h3.4L16 8.5h1A2.5 2.5 0 0 1 19.5 11v5A2.5 2.5 0 0 1 17 18.5H7A2.5 2.5 0 0 1 4.5 16v-5A2.5 2.5 0 0 1 7 8.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M12 11.2a2.6 2.6 0 0 1 2.6 2.6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="m14.7 10.3-.1 2.1 2 .3"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 15.8a2.6 2.6 0 0 1-2.6-2.6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="m9.3 16.7.1-2.1-2-.3"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function VoiceIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none">
      <rect x="9" y="4.5" width="6" height="10" rx="3" stroke="currentColor" strokeWidth="1.8" />
      <path d="M6.5 11.5a5.5 5.5 0 0 0 11 0M12 17v2.5M9.5 19.5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="3.1" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 4.5v1.8M12 17.7v1.8M19.5 12h-1.8M6.3 12H4.5M17.3 6.7l-1.3 1.3M8 16l-1.3 1.3M17.3 17.3 16 16M8 8 6.7 6.7"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M12 2.8a1.1 1.1 0 0 1 1.1 1.1v.6a7.7 7.7 0 0 1 2 .8l.4-.4a1.1 1.1 0 0 1 1.6 0l.9.9a1.1 1.1 0 0 1 0 1.6l-.4.4c.35.63.62 1.3.78 2h.62a1.1 1.1 0 0 1 1.1 1.1v1.2a1.1 1.1 0 0 1-1.1 1.1h-.6a7.7 7.7 0 0 1-.8 2l.4.4a1.1 1.1 0 0 1 0 1.6l-.9.9a1.1 1.1 0 0 1-1.6 0l-.4-.4a7.7 7.7 0 0 1-2 .8v.6a1.1 1.1 0 0 1-1.1 1.1h-1.2a1.1 1.1 0 0 1-1.1-1.1v-.6a7.7 7.7 0 0 1-2-.8l-.4.4a1.1 1.1 0 0 1-1.6 0l-.9-.9a1.1 1.1 0 0 1 0-1.6l.4-.4a7.7 7.7 0 0 1-.8-2h-.6A1.1 1.1 0 0 1 2.8 13v-1.2a1.1 1.1 0 0 1 1.1-1.1h.6a7.7 7.7 0 0 1 .8-2l-.4-.4a1.1 1.1 0 0 1 0-1.6l.9-.9a1.1 1.1 0 0 1 1.6 0l.4.4a7.7 7.7 0 0 1 2-.8V3.9A1.1 1.1 0 0 1 10.9 2.8H12Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
