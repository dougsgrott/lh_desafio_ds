

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# Function to check for missing periods in the dataset
def check_missing_periods(dataframe, category=None, time_col='ds', freq='MS'):
    if category:
        missing_periods = {}
        for cat, group in dataframe.groupby(category):
            date_range = pd.date_range(start=group[time_col].min(), end=group[time_col].max(), freq=freq)
            missing_dates = date_range.difference(group[time_col])
            missing_periods[cat] = missing_dates
        return missing_periods
    else:
        date_range = pd.date_range(start=dataframe[time_col].min(), end=dataframe[time_col].max(), freq=freq)
        missing_dates = date_range.difference(dataframe[time_col])
        return missing_dates


# Diagnostic print
def print_product_date_ranges(df, time_col, target_cols):
    for product in df['PRODUCT_NAME'].unique():
        product_df = df[df['PRODUCT_NAME'] == product]
        print(f"Product: {product}")
        print(f"  Start Date: {product_df[time_col].min()}")
        print(f"  End Date: {product_df[time_col].max()}")
        print(f"  Total Days: {len(product_df)}")
        print(f"  Days with Sales: {len(product_df[product_df[target_cols] > 0])}")
        print()


def fill_timeseries_gaps(
    df: pd.DataFrame, 
    time_col: str,
    target_cols: str,
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fill missing dates for each unique time series with zero-quantity entries.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time_col and other identifying columns
        start_date (Optional[str]): Global start date for all time series (YYYY-MM-DD)
        end_date (Optional[str]): Global end date for all time series (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: DataFrame with continuous time series
    """
    # Identify columns that define unique time series
    id_columns = [col for col in df.columns if col not in [target_cols, time_col]]
    
    # Convert time_col to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Determine global start and end dates if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    # Function to process each unique time series
    def process_timeseries(group: pd.DataFrame) -> pd.DataFrame:
        # Determine start and end dates
        group_start = start_date if start_date else group[time_col].min()
        group_end = end_date if end_date else group[time_col].max()
        
        # Create full date range
        full_date_range = pd.date_range(start=group_start, end=group_end, freq='D')
        
        # Create a template DataFrame with all dates
        continuous_df = pd.DataFrame({
            time_col: full_date_range
        })
        
        # Add identifying columns
        for col in id_columns:
            continuous_df[col] = group[col].iloc[0]
        
        # Merge with original data, filling missing dates with 0 ORDER_QTY
        merged_df = continuous_df.merge(
            group, 
            on=id_columns + [time_col], 
            how='left'
        ).fillna({
            'ORDER_QTY': 0
        })
        
        return merged_df
    
    # Group by unique time series and process
    filled_series = df.groupby(id_columns, group_keys=False).apply(process_timeseries)
    
    # Reset index and return
    return filled_series.reset_index(drop=True)



# Function to resample to a specific frequency ("M" for monthly, "W" for weekly)
def resample_time_series(df, time_col, target_cols, frequency="M"):
    # Ensure the DataFrame is sorted by date
    df = df.sort_values(time_col)
    
    # Identify columns that define unique time series
    id_columns = [col for col in df.columns if col not in [target_cols, time_col]]
    
    # Use groupby with resampling in a more concise approach
    resampled = (
        df.set_index(time_col)
        .groupby([pd.Grouper(freq=frequency)] + id_columns)[target_cols]
        .sum()
        .reset_index()
    )
    
    return resampled


# df.groupby(["STORE_NAME", "PRODUCT_NAME", pd.Grouper(key="ORDER_DATE", freq=frequency)])



# Source: pergunta8_nixtla
def fill_missing_dates(group, start_date=None, end_date=None):
    # Drop duplicates to ensure unique ORDER_DATE
    group = group.drop_duplicates(subset='ORDER_DATE')
    
    # Determine start and end dates
    start = start_date if start_date else group['ORDER_DATE'].min()
    end = end_date if end_date else group['ORDER_DATE'].max()
    
    # Create full date range
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Prepare template DataFrame
    template = pd.DataFrame({'ORDER_DATE': date_range})
    template['STORE_NAME'] = group['STORE_NAME'].iloc[0]
    template['PRODUCT_NAME'] = group['PRODUCT_NAME'].iloc[0]
    template['ORDER_QTY'] = 0
    
    # Merge original data with template
    merged = template.merge(group, on=['ORDER_DATE', 'STORE_NAME', 'PRODUCT_NAME'], how='left')
    
    # Fill missing ORDER_QTY with 0
    merged['ORDER_QTY_x'] = merged['ORDER_QTY_x'].fillna(0)
    
    # Prefer non-zero quantities
    merged['ORDER_QTY'] = np.where(
        merged['ORDER_QTY_y'] > 0, 
        merged['ORDER_QTY_y'], 
        merged['ORDER_QTY_x']
    )
    
    # Select and rename columns
    result = merged[['ORDER_DATE', 'STORE_NAME', 'PRODUCT_NAME', 'ORDER_QTY']]
    
    return result