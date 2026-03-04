import numpy as np
import pandas as pd


def normalize_data(df):
    """Perform min-max normalization on the DataFrame"""
    return (df - df.min()) / (df.max() - df.min())


def handle_outliers(df, threshold=1.5):
    """Remove outliers from the DataFrame based on IQR"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[~((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)]
    return filtered_df

