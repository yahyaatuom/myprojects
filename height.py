# this program would conver the following units to other units
# 1ft = 30.48cm
#1 inch = 2.54cm
#1ft = 12 inches
# the program would also convert the values vice versa

#Feet to inch
def feet_to_inch(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")
    else:
        return x * 12
    
#Inch to feet
def inch_to_feet(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")

    else:
        return x/12 #1 inch is equal to 1/12th of a foot
    
def cm_to_inch(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")
    else:
        return x * 2.54
    
def inch_to_cm(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")
    else:
        return x/2.54
    
def cm_to_ft(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")
    
    else:
        return x/30.48 
    
def ft_to_cm(x):
    if x <= 0:
        print("Height cannot be 0 or a negative value")

    else:
        return x * 30.48
    
height = input("Enter the user's height")