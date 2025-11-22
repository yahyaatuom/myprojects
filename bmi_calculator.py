weight = float(input(" enter your Weight in kg:"))
height = float(input("enter your height in meters:"))

bmi = weight / (height **2)
bmi = round(bmi,2)

print(f"Your BMI(Body Mass Index is: {bmi})")

if bmi < 18.5:
    print("You are underweight")
elif 18.5 <= bmi < 24.9:
    print("You have a normal weight ")   
elif 25 <= bmi < 29.9:
    print("You are overweight")
else:
    print("You are obese")         