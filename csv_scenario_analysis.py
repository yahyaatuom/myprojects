import json
import csv
from collections import Counter

file_path = r"C:\Users\Dell\Desktop\Causal-Guard\data\json\scenarios.json"

def process_scenarios(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scenarios = data.get("scenarios", [])
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON.")
        return
    print (f"Successfully loaded {len(scenarios)} scenarios.")

    return scenarios

def analyze_scenarios(scenarios):
    if not scenarios:
        return
    categories = [s.get("category", "N/A") for s in scenarios]
    category_counts = Counter(categories)

    longest_scenario = max(scenarios, key=lambda x: len(x.get("description", "")))
    longest_desc_len = len(longest_scenario.get("description", ""))

    unique_locations = set()
    for s in scenarios:
        locations_list = s.get("context", {}).get("locations", [])
        for loc in locations_list:
            name = loc.get("name")
            if name:
                unique_locations.add(name)
    print("\n--- ANALYSIS SUMMARY---")
    print(f"Categories: {dict(category_counts)}")
    print(f"Longest Description ID: {longest_scenario.get('id')} ({longest_desc_len} chars)")
    print(f"Unique Locations: {len(unique_locations)}")

    return category_counts, unique_locations

def export_to_csv(category_counts, output_file="summary.csv"):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Count"])
            for category, count in category_counts.items():
                writer.writerow([category, count])
        print(f"Summary successully exported to {output_file}")
    except Exception as e:
        print(f"Export failed: {e}")
if __name__ == "__main__":
    scenarios_list = process_scenarios(file_path)

    if scenarios_list:
        cat_counts, locations = analyze_scenarios(scenarios_list)
        export_to_csv(cat_counts)