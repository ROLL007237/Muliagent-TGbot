import os
import json
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
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

# Хранилище напоминаний
reminders = []
reminders_file = "reminders.json"

# Инициализация локальной модели Gemma 3 через Ollama
llm = ChatOllama(
    model="gemma3:27b",
    temperature=0,
    num_predict=1024,
    keep_alive="10m",
)

# Промпт для распознавания любых напоминаний
reminder_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты - помощник, который извлекает информацию о напоминаниях из текста на русском языке.
    Текущее время: {current_time}
    Текущая дата: {current_date}
    
    Проанализируй сообщение пользователя и верни ТОЛЬКО JSON объект с полями:
    
    - type: тип напоминания (meeting/deadline/call/task/shopping/health/birthday/other)
    - title: краткое название (обязательно)
    - datetime: когда напомнить (формат ГГГГ-ММ-ДД ЧЧ:ММ)
    
    Примеры:
    "встреча с Артемом завтра в 15:00" -> {{"type": "meeting", "title": "Встреча с Артемом", "datetime": "2024-01-16 15:00"}}
    "позвонить маме через 2 часа" -> {{"type": "call", "title": "Позвонить маме", "datetime": "2024-01-15 18:30"}}
    "купить хлеб сегодня вечером" -> {{"type": "shopping", "title": "Купить хлеб", "datetime": "2024-01-15 19:00"}}
    
    Верни ТОЛЬКО JSON, без пояснений."""),
    ("human", "{text}")
])

def load_reminders():
    """Загружает сохраненные напоминания"""
    global reminders
    try:
        if os.path.exists(reminders_file):
            with open(reminders_file, 'r', encoding='utf-8') as f:
                reminders = json.load(f)
                # Конвертируем строки времени обратно в datetime
                for r in reminders:
                    r['datetime'] = datetime.fromisoformat(r['datetime'])
            print(f"📅 Загружено {len(reminders)} напоминаний")
        else:
            reminders = []
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        reminders = []

def save_reminders():
    """Сохраняет напоминания в файл"""
    reminders_to_save = []
    for r in reminders:
        r_copy = r.copy()
        r_copy['datetime'] = r['datetime'].isoformat()
        reminders_to_save.append(r_copy)
    
    with open(reminders_file, 'w', encoding='utf-8') as f:
        json.dump(reminders_to_save, f, ensure_ascii=False, indent=2)

def parse_reminder_request(text):
    """Использует локальную Gemma 3 для понимания запроса"""
    try:
        current_time = datetime.now().strftime("%H:%M")
        current_date = datetime.now().strftime("%Y-%m-%d")
        chain = reminder_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "text": text,
            "current_time": current_time,
            "current_date": current_date
        })
        
        # Очищаем результат
        result = result.strip()
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        elif '```' in result:
            result = result.split('```')[1].split('```')[0].strip()
        
        data = json.loads(result)
        print(f"🔍 Распознано: {data}")
        return data
    except Exception as e:
        print(f"❌ Ошибка парсинга: {e}")
        return None

def add_reminder_from_text(text):
    """Добавляет напоминание на основе текста"""
    global reminders
    data = parse_reminder_request(text)
    
    if not data:
        return None, "❌ Не удалось понять запрос. Попробуй: 'встреча с Артемом завтра в 15:00'"
    
    if 'title' not in data or 'datetime' not in data:
        return None, "❌ Не указано название или время напоминания"
    
    try:
        reminder_time = datetime.fromisoformat(data['datetime'])
        now = datetime.now()
        
        # Если время прошло, переносим на завтра
        if reminder_time < now:
            reminder_time = reminder_time.replace(day=now.day + 1)
    except:
        return None, "❌ Неверный формат даты и времени"
    
    # Эмодзи по типу
    emoji = {
        'meeting': '📅', 'deadline': '⏰', 'call': '📞',
        'task': '✅', 'shopping': '🛒', 'health': '💊',
        'birthday': '🎂', 'other': '📌'
    }.get(data.get('type', 'other'), '📌')
    
    reminder_text = f"{emoji} {data['title']}"
    
    reminder = {
        'id': str(int(time.time() * 1000)),
        'title': data['title'],
        'datetime': reminder_time,
        'text': reminder_text,
        'notified': False
    }
    
    reminders.append(reminder)
    save_reminders()
    
    time_str = reminder_time.strftime("%d.%m.%Y в %H:%M")
    return reminder, f"✅ Запланировано {time_str}:\n{reminder_text}"

def start(update, context):
    update.message.reply_text(
        "👋 **Привет! Я бот для напоминаний!**\n\n"
        "Просто напиши мне, например:\n"
        "• 'встреча с Артемом завтра в 15:00'\n"
        "• 'позвонить маме через 2 часа'\n"
        "• 'купить хлеб сегодня вечером'\n\n"
        "Команды:\n"
        "/list - показать все напоминания\n"
        "/today - показать на сегодня",
        parse_mode='Markdown'
    )

def handle_message(update, context):
    text = update.message.text
    print(f"📨 Получено: {text}")
    
    reminder, response = add_reminder_from_text(text)
    update.message.reply_text(response)

def list_reminders(update, context):
    if not reminders:
        update.message.reply_text("📭 Нет напоминаний")
        return
    
    now = datetime.now()
    text = "📋 **Активные напоминания:**\n\n"
    
    for r in reminders:
        if not r.get('notified', False):
            time_diff = r['datetime'] - now
            if time_diff.total_seconds() > 0:
                minutes = int(time_diff.total_seconds() / 60)
                time_str = r['datetime'].strftime("%d.%m %H:%M")
                text += f"• {time_str} (через {minutes} мин) - {r['text']}\n"
    
    update.message.reply_text(text, parse_mode='Markdown')

def today_reminders(update, context):
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    today_end = today_start + timedelta(days=1)
    
    today_items = [r for r in reminders if not r.get('notified', False) 
                  and today_start <= r['datetime'] < today_end]
    
    if not today_items:
        update.message.reply_text("📭 На сегодня ничего нет")
        return
    
    text = "📅 **На сегодня:**\n\n"
    for r in sorted(today_items, key=lambda x: x['datetime']):
        text += f"• {r['datetime'].strftime('%H:%M')} - {r['text']}\n"
    
    update.message.reply_text(text, parse_mode='Markdown')

def check_reminders():
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    while True:
        try:
            now = datetime.now()
            for r in reminders:
                if not r.get('notified', False) and now >= r['datetime']:
                    bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"🔔 **НАПОМИНАНИЕ!**\n\n{r['text']}",
                        parse_mode='Markdown'
                    )
                    r['notified'] = True
                    save_reminders()
                    print(f"✅ Напоминание: {r['text']}")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            time.sleep(5)

def main():
    print("🚀 Запуск бота...")
    load_reminders()
    
    reminder_thread = threading.Thread(target=check_reminders, daemon=True)
    reminder_thread.start()
    
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("list", list_reminders))
    dp.add_handler(CommandHandler("today", today_reminders))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    
    print("✅ Бот запущен!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()