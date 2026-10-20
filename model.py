import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(path):
    return pd.read_csv(path)

def prepare_dataset(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["holiday"] = df["holiday"].map({"Yes": 1, "No": 0})
    df["weather_score"] = df["weather"].map({"Sunny": 3, "Cloudy": 2, "Rainy": 1})
    return df

def train_model(df):
    features = ["day_of_week", "month", "holiday", "weather_score", "inventory", "supply"]
    X = df[features]
    y = df["demand"]

    model = LinearRegression()
    model.fit(X, y)
    return model

def estimate_surplus(supply, predicted_demand):
    return max(supply - predicted_demand, 0)

def build_recommendations(df):
    table = []
    for _, row in df.iterrows():
        surplus = max(row["supply"] - row["demand"], 0)
        if surplus > 0:
            table.append({
                "restaurant": row["restaurant"],
                "food_type": row["food_type"],
                "supply": row["supply"],
                "demand": row["demand"],
                "surplus": round(surplus, 2),
                "recommendation": "Redistribute to local shelter or community kitchen"
            })
    return pd.DataFrame(table)
