import csv
import random

# Seed for reproducibility (optional)
random.seed(42)

# Pools for generating realistic random data
FIRST_NAMES = [
    "Yahya", "Huzaifa", "Ahmed", "Khaled", "Wasim", "Christopher", "Trump", "Ayesha", 
    "Fatema", "Salama", "Mehmood", "Sultan", "Salman", "Susan", "Joseph", 
    "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", 
    "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
]

LAST_NAMES = [
    "Ali", "Khan", "Ahmed", "Zayed", "Qassem", "Garcia", "Miller", "Davis", 
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", 
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", 
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

DEPARTMENTS = [
    "Computer Science", "Data Science", "Electrical Engineering", 
    "Mechanical Engineering", "Civil Engineering", "Mathematics", 
    "Physics", "Chemistry", "Biology", "Economics", 
    "Business Administration", "Psychology", "History", "Literature"
]

def generate_student_csv(output_file="students.csv", total_records=10000):
    # 1. Create unique sequential IDs from 0 to 9999
    # This guarantees exactly 10k unique IDs matching the '0000-9999' constraint
    student_ids = [f"{i:04d}" for i in range(total_records)]
    
    # Shuffle the IDs so they don't appear in order inside the CSV
    random.shuffle(student_ids)

    print(f"Generating {total_records} mock student records...")

    # 2. Write rows to CSV file
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow(["ID", "Name", "Department", "GPA", "Age"])
        
        for student_id in student_ids:
            # Construct names and fields
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            department = random.choice(DEPARTMENTS)
            
            # Generate GPA between 2.00 and 4.00 rounded to 2 decimal places
            gpa = round(random.uniform(2.00, 4.00), 2)
            
            # Generate typical college age distribution between 18 and 25
            age = random.randint(18, 25)
            
            writer.writerow([student_id, name, department, gpa, age])

    print(f"Success! Mock dataset saved to: '{output_file}'")

if __name__ == "__main__":
    generate_student_csv()