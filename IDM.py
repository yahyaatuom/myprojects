#Inventory data management. calculating total worth of stock at hand and also quantity of stock available
inventory = {
    "processor": {"quantity": 10, "price": 250},
    "motherboard": {"quantity": 3, "price": 150},
    "memory": {"quantity": 20, "price": 80},
    "storage": {"quantity": 15, "price": 100},
}

def calculate_total_worth(data):
    return sum(item["quantity"] * item["price"] for item in data.values())

def get_low_stock(data):
    return {name for name, details in data.items() if details["quantity"] < 3}

total_worth = calculate_total_worth(inventory)
low_stock_items = get_low_stock(inventory)
print(f"Total worth of inventory: ${total_worth}")
print(f"Items low in stock: {', '.join(low_stock_items) if get_low_stock(inventory) else 'None'}")
