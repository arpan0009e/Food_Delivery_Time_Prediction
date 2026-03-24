import pandas as pd

def load_and_preprocess(path):
    try:
        # Load dataset safely
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
        print("✅ Dataset loaded successfully")

    except Exception as e:
        print("❌ Error loading file:", e)
        return None

    # -------------------------------
    # Handle Missing Values Safely
    # -------------------------------
    
    # Fill categorical columns (only if they exist)
    for col in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fill numerical column
    if 'Courier_Experience_yrs' in df.columns:
        df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(
            df['Courier_Experience_yrs'].mean()
        )

    # -------------------------------
    # Drop unnecessary column
    # -------------------------------
    if 'Order_ID' in df.columns:
        df = df.drop('Order_ID', axis=1)

    # -------------------------------
    # Convert categorical → numerical
    # -------------------------------
    df = pd.get_dummies(df, drop_first=True)

    print("✅ Data preprocessing completed")

    return df


# Test run (optional)
if __name__ == "__main__":
    df = load_and_preprocess('../data/delivery_data.csv')
    print(df.head())