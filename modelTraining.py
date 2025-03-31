from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Train and test the model
def train_model(X_train, y_train, X_test, y_test, feature_names, model_name='rf'):
    # Initialize and fit the model
    if model_name == 'rf':
        model = RandomForestRegressor(
            verbose=1,
            n_jobs=-1,
            n_estimators=50, #try 50 # was 100
            #max_depth = 10, #default is none
            #max_features=3, # Fewer features per split, less memory
            random_state=42
        )
    else:
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=1 
        )

    model.fit(X_train, y_train)

    # Test predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    residual_vs_predicted_plot(y_test, y_pred)
    actual_vs_predicted_plot(y_test, y_pred)
    metrics = calculate_performance_metrics(y_test, y_pred)
    print_performance_metrics(metrics)
    feature_importance(model, X_train, feature_names)

    return model

# Calculates metrics
def calculate_performance_metrics(y_test, y_pred):
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    metrics['mse'] = mean_squared_error(y_test, y_pred)   # Mean Squared Error
    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)  # Mean Absolute Percentage Error
    metrics['r2'] = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)
    
    return metrics

# Prints metrics
def print_performance_metrics(metrics):
    print("Mean Absolute Error (MAE):", metrics.get('mae', "Not computed"))
    print("Mean Squared Error (MSE):", metrics.get('mse', "Not computed"))
    print("Mean Absolute Percentage Error (MAPE):", metrics.get('mape', "Not computed"))
    print("R-squared (R²):", metrics.get('r2', "Not computed"))

# Determine the feature importance in the model
def feature_importance(model, X, feature_names):
    feature_importances = model.feature_importances_
    feature_importances_list = [(feature_names[j], importance) for j, importance in enumerate(feature_importances)]
    feature_importances_list.sort(key=lambda x: x[1], reverse=True)

    print("Feature Importances:")
    for feature, importance in feature_importances_list:
        print(f"{feature}: {importance}")

# Making a plot for the presentation
def residual_vs_predicted_plot(y_test_actuals, y_pred_actuals_model, bins=70):
    y_test_actuals = pd.Series(y_test_actuals)
    y_pred_actuals_model = pd.Series(y_pred_actuals_model)

    # Calculate residuals
    residuals = y_test_actuals - y_pred_actuals_model

    # Create histogram
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=bins, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()

def actual_vs_predicted_plot(y_test_actuals, y_pred_actuals_model):
    plt.scatter(y_test_actuals, y_pred_actuals_model)
    plt.plot([min(y_test_actuals), max(y_test_actuals)], [min(y_test_actuals), max(y_test_actuals)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Volatility')
    plt.show()

# Organize the data differently
def pivot_df(df):
    df = df.copy()
    df["row_in_group"] = df.groupby(["realized_vol", "expiration"]).cumcount()
    df = df[df["row_in_group"] < 50].copy()

    df_pivot = df.set_index(["realized_vol", "expiration", "row_in_group"])
    df_pivot = df_pivot.unstack("row_in_group")

    df_pivot.columns = [
    f"{col_name}_{idx}" for col_name, idx in df_pivot.columns
    ]

    # Move the multi-level row index back into columns
    df_pivot.reset_index(inplace=True)

    return df_pivot

# Trains the model
def create_model():
    # Load the data
    processed_data = pd.read_csv('trainingData.csv')
    processed_data.drop(columns=['date', 'call'], inplace=True)

    processed_data = pivot_df(processed_data)

    # Separate data
    X = processed_data.drop(columns=['realized_vol']).values
    y = processed_data['realized_vol'].values

    # Scaling all features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For the feature importance
    feature_names = processed_data.columns[:-1].tolist()

    # Get test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the model
    model = train_model(X_train, y_train, X_test, y_test, feature_names, model_name='xg')

    return model

file_name = "model.pkl"
pickle.dump(create_model(), open(file_name, "wb"))