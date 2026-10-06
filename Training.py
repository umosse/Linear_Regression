import pandas as pd
import sys
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame | None:
    """
    Takes a path as argument and returns a data set after checking if its content is valid.
    """

    try:
        data_set = pd.read_csv(path)

    #Check for known file reading errors
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        print("Error loading file at", path, err, file=sys.stderr)
        return None

    #Check for other unexpected errors
    except Exception as err:
        print("An unexpected error occurred", err, file=sys.stderr)
        return None

    #Check if there is a 'km' and a 'price' column
    if not {"km", "price"}.issubset(data_set.columns):
        print("Dataset must have a 'km' and a 'price' column", file=sys.stderr)
        return None

    data_set = data_set[["km", "price"]]

    #Check if 'km' and 'price' are numeric
    if not all(pd.api.types.is_numeric_dtype(data_set[x]) for x in data_set.columns):
        print("'km' and 'price' must be numeric values", file=sys.stderr)
        return None

    #Check if there are no missing values in columns
    if data_set.isna().any().any():
        print("Dataset has missing values", file=sys.stderr)
        return None

    if len(data_set) < 2 or data_set["km"].nunique() < 2:
        print("There must be at least 2 rows with different mileages", file=sys.stderr)
        return None

    return data_set


def estimate_price(theta0, theta1, mileage) -> float:
    """
    Predict the price of a car for a given mileage using theta0 and theta1
    """
    return theta0 + (theta1 * mileage)


def gradient_step(km, price, theta0, theta1, lr) -> tuple[float, float]:
    """

    """
    m = len(km)
    errors = [estimate_price(theta0, theta1, km[i]) - price[i] for i in range(m)]

    sum_theta0 = sum(errors)
    sum_theta1 = sum(errors[i] * km[i] for i in range(m))

    tmp_theta0 = lr * sum_theta0 / m
    tmp_theta1 = lr * sum_theta1 / m

    return tmp_theta0, tmp_theta1


def normalize(km) -> list[float]:
    """
    """
    low = min(km)
    high = max(km)
    return [(x - low) / (high - low) for x in km]


def denormalize(theta0_norm, theta1_norm, low, high) -> tuple[float, float]:
    """
    Convert normalized thetas back to regular mileage scale
    """
    km_range = high - low
    theta1 = theta1_norm / km_range
    theta0 = theta0_norm - theta1_norm * low / km_range
    return theta0, theta1


def train(km, price, lr, iterations) -> tuple[float, float]:
    """
    Run gradient descent on (km price) and return de-normalized theta0 and theta1
    """
    low = min(km)
    high = max(km)
    normalized = normalize(km)

    theta0 = 0.0
    theta1 = 0.0

    for x in range(iterations):
        tmp0, tmp1 = gradient_step(normalized, price, theta0, theta1, lr)
        theta0 -= tmp0
        theta1 -= tmp1
        # if x % 100 == 0:
        #     errors = [estimate_price(theta0, theta1, normalized[y]) - price[y] for y in range(len(normalized))]
        #     cost = sum(e**2 for e in errors) / (2 * len(normalized))
        #     print(x, cost)

    return denormalize(theta0, theta1, low, high)


def save_thetas(theta0, theta1, path) -> None:
    """
    Saves the thetas to a .csv file
    """
    pd.DataFrame([{"theta0": theta0, "theta1": theta1}]).to_csv(path, index=False)


def plot_data(km, price, theta0, theta1) -> None:
    """
    Plots the dataset as a scatterplot with the regression line over it
    """

    x = [min(km), max(km)]
    y = [estimate_price(theta0, theta1, i) for i in x]

    plt.scatter(km, price, label="Data")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Car prices depending on mileage")
    plt.plot(x, y, label="Regression line", color="red")
    plt.legend()
    plt.show()


def main():
    data_set = load("data.csv")
    if data_set is None:
        sys.exit(1)
    theta0, theta1 = train(data_set["km"], data_set["price"], lr=0.1, iterations=1000)
    save_thetas(theta0, theta1, "thetas.csv")
    print(theta0, theta1)
    plot_data(data_set["km"], data_set["price"], theta0, theta1)


if __name__ == "__main__":
    main()
