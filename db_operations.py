import sqlite3

# Connect to the SQLite3 database
conn = sqlite3.connect('user_data.db')
cursor = conn.cursor()

# Function to add a new user
def add_user(username, hashed_password):
    try:
        cursor.execute('INSERT INTO users (username, hashed_password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        print(f"User {username} added successfully.")
    except sqlite3.IntegrityError:
        print(f"Error: Username {username} already exists.")

# Function to retrieve user data
def get_user(username):
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    return cursor.fetchone()

# Function to update user password
def update_user_password(username, new_hashed_password):
    cursor.execute('UPDATE users SET hashed_password = ? WHERE username = ?', (new_hashed_password, username))
    conn.commit()
    print(f"Password for {username} updated successfully.")

# Function to delete a user
def delete_user(username):
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    print(f"User {username} deleted successfully.")

# Close the database connection when done
conn.close() 