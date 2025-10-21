# PySpark Analysis Project

Проект для аналізу даних з використанням Apache Spark.

## Використання

**Запуск:**

```bash
# Запустити контейнер (перший раз)
docker-compose up --build -d

# Запустити контейнер (наступні рази)
docker-compose up -d

# Виконати main.py
docker exec -it pyspark_app python3 -m src.main
```

**Зупинка:**

```bash
docker-compose down
```

**Інтерактивна робота:**

```bash
docker exec -it pyspark_app bash
```

**Перебудувати після змін в Dockerfile:**

```bash
docker-compose up --build -d
```

## Структура

```
├── data/              # Дані (автоматично монтується в /app/data)
├── src/               # Python код (автоматично монтується в /app/src)
│   └── main.py
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml     # Залежності проекту
```
