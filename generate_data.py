"""Create the three practice tables used in the notebook (reproducible, seed = 42).

Run:  python generate_data.py
Output: data/customers.csv, data/products.csv, data/orders.csv

The data is SYNTHETIC and intentionally a little messy (missing ages, mixed-case
cities, duplicate orders, dates stored as text) so the notebook has real cleaning
work to do.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 400
N_ORDERS = 3000
DATA_DIR = Path(__file__).resolve().parent / "data"

CITIES = ["Jaipur", "Delhi", "Mumbai", "Bengaluru", "Pune", "Kolkata", "Hyderabad", "Ahmedabad"]
PRODUCTS = [  # (name, category, unit_price, cost_price) in INR
    ("Wireless Earbuds", "Electronics", 1999, 1100), ("Power Bank", "Electronics", 1499, 850),
    ("Smartwatch", "Electronics", 3499, 2200), ("Bluetooth Speaker", "Electronics", 2499, 1400),
    ("Cotton T-Shirt", "Fashion", 599, 250), ("Denim Jeans", "Fashion", 1599, 800),
    ("Running Shoes", "Fashion", 2799, 1500), ("Cotton Kurta", "Fashion", 999, 450),
    ("Mixer Grinder", "Home & Kitchen", 3299, 2000), ("Air Fryer", "Home & Kitchen", 4999, 3100),
    ("Cookware Set", "Home & Kitchen", 2199, 1200), ("Cotton Bedsheet", "Home & Kitchen", 899, 400),
    ("Face Serum", "Beauty", 799, 300), ("Matte Lipstick", "Beauty", 499, 180),
    ("Sunscreen SPF 50", "Beauty", 449, 170), ("Hair Oil", "Beauty", 349, 130),
    ("Python Programming Book", "Books", 699, 350), ("Statistics Textbook", "Books", 899, 450),
    ("Fiction Novel", "Books", 399, 180), ("Notebook Set", "Books", 299, 120),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    # ---- customers ----
    customers = pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(N_CUSTOMERS)],
        "city": rng.choice(CITIES, N_CUSTOMERS, p=[0.10, 0.18, 0.18, 0.16, 0.12, 0.10, 0.10, 0.06]),
        "age": rng.integers(18, 61, N_CUSTOMERS).astype(float),
        "gender": rng.choice(["Female", "Male"], N_CUSTOMERS, p=[0.52, 0.48]),
    })
    customers.loc[rng.choice(N_CUSTOMERS, 24, replace=False), "age"] = np.nan        # missing ages
    messy = rng.choice(N_CUSTOMERS, 30, replace=False)
    customers.loc[messy, "city"] = customers.loc[messy, "city"].str.upper()          # inconsistent case

    # ---- products ----
    products = pd.DataFrame(PRODUCTS, columns=["product_name", "category", "unit_price", "cost_price"])
    products.insert(0, "product_id", [f"P{100 + i}" for i in range(len(products))])

    # ---- orders (festive season boost in Oct-Nov) ----
    month_weights = np.array([6, 6, 7, 7, 7, 6, 6, 7, 8, 13, 14, 9], dtype=float)
    month_weights /= month_weights.sum()
    months = rng.choice(np.arange(1, 13), N_ORDERS, p=month_weights)
    days = rng.integers(1, 29, N_ORDERS)
    dates = pd.to_datetime({"year": 2025, "month": months, "day": days})
    discount = rng.choice([0, 5, 10, 15, 20], N_ORDERS, p=[0.30, 0.20, 0.22, 0.16, 0.12])
    quantity = 1 + rng.poisson(0.35 + 0.03 * discount)
    orders = pd.DataFrame({
        "order_id": [f"O{50000 + i}" for i in range(N_ORDERS)],
        "customer_id": rng.choice(customers["customer_id"], N_ORDERS),
        "product_id": rng.choice(products["product_id"], N_ORDERS),
        "order_date": dates.dt.strftime("%Y-%m-%d"),                                # stored as text
        "quantity": np.clip(quantity, 1, 6),
        "discount_pct": discount,
        "payment_method": rng.choice(["UPI", "Card", "COD", "Wallet"], N_ORDERS, p=[0.48, 0.27, 0.17, 0.08]),
    })
    orders = pd.concat([orders, orders.sample(15, random_state=SEED)], ignore_index=True)  # duplicates

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    print(f"customers: {customers.shape} | products: {products.shape} | orders: {orders.shape}")


if __name__ == "__main__":
    main()
