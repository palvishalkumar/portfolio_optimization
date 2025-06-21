import bcrypt

# List of plain text passwords
passwords = ["12345", "admin123"]

# Generate and print hashed passwords
hashed_passwords = [bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() for password in passwords]

print("Hashed Passwords:")
print(hashed_passwords)
