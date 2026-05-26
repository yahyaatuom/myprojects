def reverse_string(s):
    reversed_text = ""
    for char in s:
        reversed_text = char + reversed_text
    return reversed_text

result = reverse_string("Hello, World!")
print(result)