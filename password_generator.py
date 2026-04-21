import random
import string

length = int(input("Enter the desired length of the password:"))

letters = string.ascii_letters
digits = string.digits
symbols = string.punctuation

all_chars = letters + digits + symbols

password = "".join(random.choice(all_chars) for _ in range(length))

print(f"Your generated Password is:  {password}")