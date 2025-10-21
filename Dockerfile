# Етап 1: Використовуємо офіційний, перевірений образ Spark як основу
FROM spark:4.0.0-scala2.13-java17-python3-ubuntu

# Етап 2: Встановлюємо uv
COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /usr/local/bin/

# Перемикаємося на root для встановлення залежностей
USER root

# Етап 3: Встановлюємо залежності
WORKDIR /app

# Спочатку копіюємо файли конфігурації
COPY pyproject.toml ./

# Встановлюємо залежності (uv згенерує lock-файл автоматично для Python 3.10)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Копіюємо решту коду
COPY ./src ./src

# Даємо права на /app для будь-якого користувача
RUN chmod -R 777 /app
