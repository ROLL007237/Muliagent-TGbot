import os
import json
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
# Импорты для LangChain (актуальные для версий 1.x)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('YOUR_CHAT_ID')

# Проверка наличия ключей
if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN не найден в .env файле!")
if not CHAT_ID:
    raise ValueError("❌ YOUR_CHAT_ID не найден в .env файле!")

# Хранилище встреч
meetings = []
meetings_file = "meetings.json"

# Инициализация локальной модели Gemma 3 через Ollama
llm = ChatOllama(
    model="gemma3:27b",
    temperature=0,
    num_predict=1024,
    top_k=10,
    top_p=0.5,
    keep_alive="10m",  # Держим модель в памяти 10 минут после последнего запроса
)

# Промпт для распознавания встреч
meeting_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты - помощник, который извлекает информацию о встречах из текста на русском языке.
    Текущее время: {current_time}
    
    Проанализируй сообщение пользователя и верни ТОЛЬКО JSON объект с полями:
    - person: с кем встреча (если указано)
    - time: время встречи в формате ЧЧ:ММ (если указано абсолютное время)
    - topic: тема встречи (если есть)
    - minutes_until: через сколько минут (если указано относительное время)
    
    Примеры:
    "поставь встречу с Артемом на 18:15" -> {{"person": "Артем", "time": "18:15"}}
    "напомни позвонить маме через 10 минут" -> {{"person": "мама", "minutes_until": 10, "topic": "позвонить"}}
    "встреча с командой завтра в 10 утра" -> {{"person": "команда", "time": "10:00", "topic": "встреча"}}
    "совещание в 14:30" -> {{"time": "14:30", "topic": "совещание"}}
    
    Верни ТОЛЬКО JSON, без пояснений."""),
    ("human", "{text}")
])

def load_meetings():
    """Загружает сохраненные встречи"""
    global meetings
    try:
        if os.path.exists(meetings_file):
            with open(meetings_file, 'r', encoding='utf-8') as f:
                meetings = json.load(f)
                # Конвертируем строки времени обратно в datetime
                for m in meetings:
                    m['time'] = datetime.fromisoformat(m['time'])
            print(f"📅 Загружено {len(meetings)} встреч")
        else:
            meetings = []
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        meetings = []

def save_meetings():
    """Сохраняет встречи в файл"""
    meetings_to_save = []
    for m in meetings:
        m_copy = m.copy()
        m_copy['time'] = m['time'].isoformat()
        meetings_to_save.append(m_copy)
    
    with open(meetings_file, 'w', encoding='utf-8') as f:
        json.dump(meetings_to_save, f, ensure_ascii=False, indent=2)

def parse_meeting_request(text):
    """Использует локальную Gemma 3 для понимания запроса"""
    try:
        current_time = datetime.now().strftime("%H:%M %d.%m.%Y")
        chain = meeting_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "text": text,
            "current_time": current_time
        })
        
        # Очищаем результат от возможных markdown-форматирований
        result = result.strip()
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        elif '```' in result:
            result = result.split('```')[1].split('```')[0].strip()
        
        # Парсим JSON
        data = json.loads(result)
        print(f"🔍 Распознано: {data}")
        return data
    except Exception as e:
        print(f"❌ Ошибка парсинга: {e}")
        print(f"   Текст ответа: {result if 'result' in locals() else 'нет ответа'}")
        return None

def add_meeting_from_text(text):
    """Добавляет встречу на основе текста"""
    global meetings
    data = parse_meeting_request(text)
    
    if not data:
        return None, "Не удалось понять запрос. Попробуй: 'встреча с Артемом в 18:15'"
    
    now = datetime.now()
    
    # Определяем время встречи
    if 'time' in data:
        # Абсолютное время
        time_str = data['time']
        try:
            # Парсим время
            meeting_time = datetime.strptime(time_str, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            # Если время уже прошло сегодня, переносим на завтра
            if meeting_time < now:
                meeting_time += timedelta(days=1)
        except:
            return None, "Неверный формат времени. Используй ЧЧ:ММ, например 18:15"
    
    elif 'minutes_until' in data:
        # Относительное время
        try:
            minutes = int(data['minutes_until'])
            meeting_time = now + timedelta(minutes=minutes)
        except:
            return None, "Неверный формат времени"
    else:
        return None, "Не указано время встречи"
    
    # Формируем текст напоминания
    person = data.get('person', '')
    topic = data.get('topic', 'встреча')
    
    if person and topic != 'встреча':
        reminder_text = f"🔔 {topic} с {person}"
    elif person:
        reminder_text = f"🔔 Встреча с {person}"
    else:
        reminder_text = f"🔔 {topic}"
    
    # Добавляем встречу
    meeting = {
        'time': meeting_time,
        'text': reminder_text,
        'notified': False,
        'created': now.isoformat(),
        'raw_request': text
    }
    meetings.append(meeting)
    save_meetings()
    
    # Формируем ответ
    minutes_until = int((meeting_time - now).total_seconds() / 60)
    if minutes_until < 60:
        time_str = f"через {minutes_until} мин"
    else:
        time_str = meeting_time.strftime("в %H:%M")
    
    response = f"✅ Запланировала {time_str}: {reminder_text}"
    return meeting, response

def start(update, context):
    """Обработчик команды /start"""
    update.message.reply_text(
        "👋 Привет! Я умный бот с локальной Gemma 3 27B!\n\n"
        "Просто напиши мне:\n"
        "• 'встреча с Артемом в 18:15'\n"
        "• 'напомни позвонить маме через 10 минут'\n"
        "• 'созвон с командой завтра в 10 утра'\n"
        "• 'совещание в 14:30'\n\n"
        "Команды:\n"
        "/list - показать все встречи\n"
        "/clear - очистить все встречи"
    )

def handle_message(update, context):
    """Обрабатывает текстовые сообщения"""
    text = update.message.text
    print(f"📨 Получено: {text}")
    
    meeting, response = add_meeting_from_text(text)
    
    if meeting:
        update.message.reply_text(response)
    else:
        update.message.reply_text(response or "❌ Не поняла запрос")

def list_meetings(update, context):
    """Показать все встречи"""
    if not meetings:
        update.message.reply_text("📭 Нет запланированных встреч")
        return
    
    now = datetime.now()
    text = "📅 **Запланированные встречи:**\n\n"
    
    # Сортируем по времени
    sorted_meetings = sorted(meetings, key=lambda x: x['time'])
    active_meetings = [m for m in sorted_meetings if not m['notified']]
    
    if not active_meetings:
        update.message.reply_text("📭 Нет активных встреч")
        return
    
    for m in active_meetings[:10]:
        minutes_until = int((m['time'] - now).total_seconds() / 60)
        if minutes_until > 0:
            time_str = m['time'].strftime("%H:%M")
            text += f"• {time_str} (через {minutes_until} мин) - {m['text']}\n"
    
    update.message.reply_text(text, parse_mode='Markdown')

def clear_meetings(update, context):
    """Очистить все встречи"""
    global meetings
    meetings = []
    save_meetings()
    update.message.reply_text("🗑️ Все встречи удалены")

def check_reminders():
    """Поток для проверки напоминаний"""
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    while True:
        try:
            now = datetime.now()
            for meeting in meetings:
                if not meeting['notified'] and now >= meeting['time']:
                    bot.send_message(
                        chat_id=CHAT_ID,
                        text=meeting['text']
                    )
                    meeting['notified'] = True
                    save_meetings()
                    print(f"✅ Напоминание: {meeting['text']} в {now.strftime('%H:%M:%S')}")
            
            time.sleep(5)
        except Exception as e:
            print(f"❌ Ошибка в потоке напоминаний: {e}")
            time.sleep(5)

def main():
    print("🚀 Запуск умного бота с локальной Gemma 3 27B...")
    
    # Загружаем сохраненные встречи
    load_meetings()
    
    # Запускаем поток проверки
    reminder_thread = threading.Thread(target=check_reminders, daemon=True)
    reminder_thread.start()
    
    # Запускаем бота
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("list", list_meetings))
    dp.add_handler(CommandHandler("clear", clear_meetings))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    
    print("✅ Бот запущен!")
    print("📱 Пиши в Telegram, например: 'встреча с Артемом в 18:15'")
    print(f"🤖 Используется локальная модель gemma3:27b через Ollama")
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()