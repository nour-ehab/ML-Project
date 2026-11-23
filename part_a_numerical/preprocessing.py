import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_and_preprocess():
    medical_df = pd.read_csv("data/insurance.csv")
    
    medical_df = medical_df.drop_duplicates()
    medical_df = medical_df.dropna(subset=["charges"])
    
    inputs = medical_df.drop(columns=["charges"])
    target = medical_df["charges"]
    
    num_cols = inputs.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = inputs.select_dtypes(include=["object", "category"]).columns.tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        inputs, target, test_size=0.30, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)
    
    num_pipes = make_pipeline(
        SimpleImputer(strategy="mean"),
        MinMaxScaler())
    
    cat_pipes = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))
    
    preprocessing = ColumnTransformer([
        ("num", num_pipes, num_cols),
        ("cat", cat_pipes, cat_cols)])

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessing