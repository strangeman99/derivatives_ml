import pickle
import pandas as pd
import json

# Getting trading information for options
def options_data():
    # Read dataset
    options_df = pd.read_csv("2024.csv")
    options_df['expiration'] = pd.to_datetime(options_df['expiration'], format="%Y-%m-%d %H:%M:%S %z UTC")
    options_df['date'] = pd.to_datetime(options_df['date'], format="%Y-%m-%d %H:%M:%S %z UTC")

    # Format dates
    options_df['date'] = options_df['date'].dt.tz_localize(None)
    options_df['expiration'] = options_df['expiration'].dt.tz_localize(None)

    # Adjust data
    options_df['expiration'] = (options_df['expiration'] - options_df['date']).dt.days
    options_df["call"] = options_df['call_put'].astype(str).str.contains("Call").astype(int)
    options_df.drop(columns=["call_put", "act_symbol", "call"], inplace=True)

    return options_df

# Create the right formatting
def pivot_df(df):
    df = df.copy()
    df["row_in_group"] = df.groupby(["date", "expiration"]).cumcount()

    # Limit the number of options in the model to 50
    df = df[df["row_in_group"] < 50].copy()

    # Sort on the right index
    df_pivot = df.set_index(["date", "expiration", "row_in_group"])
    df_pivot = df_pivot.unstack("row_in_group")
    df_pivot.columns = [
    f"{col_name}_{idx}" for col_name, idx in df_pivot.columns
    ]

    # Move the multi-level row index back into columns
    df_pivot.reset_index(inplace=True)

    return df_pivot

def main():
    # Model
    file_name = "model.pkl"
    xgb_model = pickle.load(open(file_name, "rb"))

    # Data
    options_df = options_data()
    formatted_df = pivot_df(options_df)

    # Get predicts
    X = formatted_df.drop("date", axis=1)
    predictions = xgb_model.predict(X)

    # Format and push to json
    results_df = pd.DataFrame()
    results_df["date"] = formatted_df["date"]
    results_df["expiration"] = formatted_df["expiration"]
    results_df["vol_pred"] = predictions

    results_json = results_df.to_json(orient="records", date_format="iso")
    with open("model_results_2024.json", "w") as f:
        f.write(results_json)

main()
