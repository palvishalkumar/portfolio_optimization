import sqlite3

def init_database():
    # Connect to SQLite3 database (or create it if it doesn't exist)
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    # Drop existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS users')

    # Create a table for storing user information with all required fields
    cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        hashed_password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_database()
    print("Database initialized successfully!") 