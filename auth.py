# # utils/auth.py

import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
import re

def validate_username(username):
    """
    Validate username format:
    - Must be 3-20 characters long
    - Can only contain letters, numbers, underscores
    - Must start with a letter
    - No spaces allowed
    """
    if not 3 <= len(username) <= 20:
        return False, "Username must be between 3 and 20 characters long"
    if not re.match("^[a-zA-Z][a-zA-Z0-9_]*$", username):
        return False, "Username must start with a letter and can only contain letters, numbers, and underscores"
    return True, "Valid username"

def init_connection():
    return sqlite3.connect('user_data.db')

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def register_user(username, name, email, password):
    # Validate username format first
    is_valid, message = validate_username(username)
    if not is_valid:
        return False, message

    # Check if username exists before attempting to insert
    conn = init_connection()
    cursor = conn.cursor()
    try:
        # Check for existing username
        cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return False, "Username already exists"
        
        # If username is available, proceed with registration
        hashed_pw = hash_password(password)
        cursor.execute(
            'INSERT INTO users (username, name, email, hashed_password) VALUES (?, ?, ?, ?)',
            (username, name, email, hashed_pw)
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError as e:
        if "email" in str(e):
            return False, "Email already registered"
        return False, "Registration failed"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"
    finally:
        conn.close()

def verify_user(username, password):
    conn = init_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT name, hashed_password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        if result and verify_password(password, result[1]):
            return True, result[0]  # Return True and the user's name
        return False, None
    finally:
        conn.close()

def get_user_data(username):
    conn = init_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT name, email FROM users WHERE username = ?', (username,))
        return cursor.fetchone()
    finally:
        conn.close()

# For backward compatibility with existing code
def get_hashed_user_data():
    conn = init_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT username, name, hashed_password FROM users')
        users = cursor.fetchall()
        
        user_data = {"usernames": {}}
        for username, name, hashed_password in users:
            user_data["usernames"][username] = {
                "name": name,
                "password": hashed_password
            }
        return user_data
    finally:
        conn.close()
