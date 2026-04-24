import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from plyer import notification

# Настройка локальной модели (дирижеру лучше дать модель помощнее)
local_llm = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

# ИНСТРУМЕНТ (для одного из агентов)
@tool("send_notification")
def send_notification(title: str, message: str):
    """Отправляет системное уведомление на компьютер."""
    notification.notify(title=title, message=message, timeout=10)
    return "Уведомление отправлено"

# 1. АГЕНТЫ-ИСПОЛНИТЕЛИ
researcher = Agent(
    role='Исследователь',
    goal='Найти информацию о погоде или событиях',
    backstory='Ты ищешь данные. Ты не умеешь отправлять уведомления.',
    llm=local_llm
)

notificator = Agent(
    role='Специалист по оповещениям',
    goal='Ставить напоминания пользователю',
    backstory='Твоя единственная работа — использовать инструмент уведомлений.',
    tools=[send_notification],
    llm=local_llm
)

analyst = Agent(
    role='Аналитик',
    goal='Группировать и проверять данные',
    backstory='Ты проверяешь, чтобы информация была краткой и точной.',
    llm=local_llm
)

# 2. ЗАДАЧА (одна общая для всей группы)
main_task = Task(
    description="Узнай, какая сегодня погода в Москве (условно), проанализируй, нужна ли куртка, и отправь мне уведомление с советом."
)

# 3. КОРПУС (ОРКЕСТР)
crew = Crew(
    agents=[researcher, analyst, notificator],
    tasks=[main_task],
    process=Process.manager,  # Включает режим оркестра
    manager_llm=local_llm,    # Назначаем дирижера (он будет управлять остальными)
    verbose=True
)

crew.kickoff()
