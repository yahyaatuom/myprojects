def max (x,y,z):
    if x > y and x >z:
        print ("The maximum number is: ", x)
    elif y > x and y > z:
        print ("The maximum number is: ", y)
    elif z > x and z > y:
        print("The maximum number is: ", z)
    else:
        print("All numbers are equal")

max(5, 1 ,54)