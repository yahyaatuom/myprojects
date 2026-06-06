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
    
height = input("Enter the user's height:")

conversion_router = {
    ('ft', 'cm'): ft_to_cm,
    ('cm', 'ft'): cm_to_ft,
    ('inch', 'cm'): inch_to_cm,
    ('cm','inch'): cm_to_inch,
    ('ft', 'inch'): feet_to_inch,
    ('inch', 'feet'): inch_to_feet,
}

def get_unit_choice(text):
    """The user would enter the unit of their choice and also mention the unit they'd like their height to be converted to"""
    valid_units = ['ft','cm','inch']
    while True:
        choice = input(text).strip().lower()
        if choice in valid_units:
            return choice
        else:
            print ("Invalid choice")

def get_float_input(text):
    """"Ensures the user enters a valid numerical value"""
    while True:
        try:
            return float(input(text))
        except ValueError:
            print("Invalid Input")

def main():
    print("=== Height conversion Tool ===")
    from_unit = get_unit_choice("Enter the input unit(ft, cm, inch)")
    value = get_float_input("Enter the value in {from_unit}:")
    to_unit = get_unit_choice("Enter the unit to convert to (ft, cm, inch):")
    conversion_function = conversion_router.get((from_unit,to_unit))
    result = conversion_function(value)
    print(f"\nResult: {value} {from_unit} = {result:.2f} {to_unit}")

if __name__ == "__main__":
    main()