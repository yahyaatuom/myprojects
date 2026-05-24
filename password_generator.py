import random
import string

length = int(input("Enter the desired length of the password:")) #length of the password to be generated

letters = string.ascii_letters # includes both uppercase and lowercase letters
digits = string.digits         #includes digits from 0 to 9
symbols = string.punctuation   # includes special characters like !, @, #, $, etc.

all_chars = letters + digits + symbols

password = "".join(random.choice(all_chars) for _ in range(length))

print(f"Your generated Password is:  {password}")