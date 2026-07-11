#defining a function that would find the minimum of numbers
def find_min(numbers):
    min_num = numbers[0]
    for num in numbers:
        if num < min_num:
            min_num = num
    return min_num

numbers = [int(x) for x in input("nums: ").split()]
print(find_min(numbers))