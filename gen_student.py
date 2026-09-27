import csv
import random
import os

def generate_dataset(n, filename):
    departments = ['CS', 'EE', 'ME', 'CE', 'BBA', 'Math', 'Physics', 'Chemistry']
    first_names = ['Yahya', 'Ahmed', 'Sara', 'Ayesha', 'Bilal', 'Hina', 'Usman', 'Fatima', 
                   'Hassan', 'Zainab', 'Omar', 'Maryam', 'Tariq', 'Noor', 'Kamran']
    last_names = ['Khan', 'Ahmed', 'Raza', 'Malik', 'Sheikh', 'Qureshi', 'Butt', 
                  'Chaudhry', 'Syed', 'Abbasi', 'Mughal', 'Ansari']
    
    records = []
    for i in range(1, n + 1):
        student_id = i
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        dept = random.choice(departments)
        gpa = round(random.uniform(2.0, 4.0), 2)
        age = random.randint(18, 26)
        records.append([student_id, name, dept, gpa, age])
    
    # Shuffle records for HEAP insertion order (random original order)
    random.shuffle(records)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Student_ID', 'Student_Name', 'Department', 'GPA', 'Age'])
        writer.writerows(records)
    
    print(f"Generated {filename} with {n} records (shuffled order)")

generate_dataset(10000, 'students_10k.csv')
generate_dataset(100000, 'students_100k.csv')