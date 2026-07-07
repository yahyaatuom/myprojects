#Lets see how to import a function from another program in the same directory
from height import feet_to_inch

height = float(input("Enter your height in feet: "))
height_in_inches = feet_to_inch(height)
print(height_in_inches)