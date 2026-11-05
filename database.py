import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "researchiq.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_document(filename, page_count, chunk_count):
    with get_connection() as connection:
        cursor = connection.execute(
            """INSERT INTO documents
            (filename, page_count, chunk_count)
            VALUES (?, ?, ?)""",
            (filename, page_count, chunk_count)
        )
        return cursor.lastrowid


def save_chat(question, answer, sources_json):
    with get_connection() as connection:
        connection.execute(
            """INSERT INTO chat_history
            (question, answer, sources_json)
            VALUES (?, ?, ?)""",
            (question, answer, sources_json)
        )


def get_documents():
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM documents ORDER BY id DESC"
        ).fetchall()


def get_chat_history(limit=50):
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
