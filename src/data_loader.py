import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

if __name__ == "__main__":
    load_data(r"C:\Users\sanje\DS-Intern-Assignment-Satwik\data\data.csv")
