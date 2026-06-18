import sqlite3

conn = sqlite3.connect(
    "database/diabetes.db"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    gender TEXT,

    age REAL,

    hypertension INTEGER,

    heart_disease INTEGER,

    smoking_history TEXT,

    bmi REAL,

    hba1c REAL,

    glucose REAL,

    probability REAL,

    prediction TEXT
)
""")

conn.commit()
conn.close()

print("Database Created")