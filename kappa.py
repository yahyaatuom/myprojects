# a code in which the user enters values and the code calculates the kappa value

from sklearn.metrics import cohen_kappa_score

def get_kappa():
    print("Enter the labels for Rater A (separated by commas):")
    input_a = input().split(',')
    print("\nEnter the labels for Rater B (seperated by commas):")
    input_b = input().split(',')

    rater_a = [label.strip() for label in input_a]
    rater_b = [label.strip() for label in input_b]

    if len(rater_a) != len(rater_b):
        return "Error: Raters must have the same number of labels."
    
    score = cohen_kappa_score(rater_a, rater_b)
    return f"Cohen's Kappa Score: {score:.4f}"

print(get_kappa())