import pandas as pd
import sys
from Training import estimate_price


def load_thetas(path) -> tuple[float, float]:
    """
    Loads the thetas from a .csv file
    Returns (0.0 0.0) if the file is not correct
    """
    try:
        df = pd.read_csv(path)
        return float(df["theta0"].iloc[0]), float(df["theta1"].iloc[0])
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return 0.0, 0.0


def main():
    try:
        theta0, theta1 = load_thetas("thetas.csv")
        mileage = float(input("Please enter a mileage: "))
        if mileage < 0: 
            raise ValueError("Mileage cannot be negative")
        prediction = estimate_price(theta0, theta1, mileage)
        print("The price for ", mileage, "is predicted to be:", prediction, "€")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
