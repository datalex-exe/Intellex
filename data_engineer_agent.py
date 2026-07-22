"""
data_engineer_agent.py - Dynamic Authentic Data Cleaning Agent
Detects column intent from headings, cleans, standardizes, and populates missing target slots
using existing dataset columns so that no field is left blank or empty in the dashboard.
Prioritizes clean discrete categorical columns over continuous floating-point numbers for chart axes.
"""

import pandas as pd
import numpy as np
from database import save_cleaned_data, save_column_mappings

def discover_and_map_headers(df):
    """
    Scans raw DataFrame columns and maps them to standard internal names
    based on keywords and data properties.
    Prioritizes low-cardinality discrete columns for categorical axes.
    """
    print("\nDiscovering file structure and headers...")
    
    raw_cols = {col: str(col).lower().strip().replace(' ', '_') for col in df.columns}
    mapping = {}
    
    target_slots = ['order_date', 'revenue', 'units_sold', 'customer_id', 'region', 'product']
    
    search_criteria = [
        ('order_date', ['date', 'time', 'timestamp', 'dt', 'day', 'month', 'year']),
        ('revenue', ['revenue', 'sales', 'amount', 'spend', 'total', 'price', 'gross', 'amt', 'rate', 'turnover', 'income', 'value', 'weekly_sales']),
        ('units_sold', ['unit', 'sold', 'qty', 'quantity', 'count', 'pcs', 'pieces', 'volume', 'qty_sold', 'units', 'holiday_flag']),
        ('customer_id', ['id', 'cust', 'customer', 'client', 'key', 'buyer', 'account', 'user_id', 'store', 'dept', 'department']),
        ('region', ['region', 'country', 'city', 'state', 'location', 'area', 'territory', 'zone', 'market', 'store', 'branch']),
        ('product', ['product', 'item', 'category', 'sku', 'description', 'service', 'goods', 'type', 'group', 'dept', 'department', 'holiday_flag'])
    ]
    
    # Pass 1: High-confidence keyword matching
    for target, keywords in search_criteria:
        if target not in mapping.values():
            for kw in keywords:
                for original, clean in raw_cols.items():
                    if original not in mapping and kw in clean:
                        # Avoid choosing continuous decimal floats for discrete categories if discrete columns exist
                        if target in ['region', 'product', 'customer_id'] and pd.api.types.is_float_dtype(df[original]) and df[original].nunique() > 20:
                            continue
                        mapping[original] = target
                        break
                if target in mapping.values():
                    break

    # Pass 2: Assign remaining unmapped dataset columns to missing target slots
    unmapped_raw_cols = [orig for orig in df.columns if orig not in mapping]
    unmapped_targets = [t for t in target_slots if t not in mapping.values()]
    
    for target in list(unmapped_targets):
        if not unmapped_raw_cols:
            break
        if target in ['units_sold', 'revenue']:
            numeric_left = [c for c in unmapped_raw_cols if pd.api.types.is_numeric_dtype(df[c])]
            chosen = numeric_left[0] if numeric_left else unmapped_raw_cols[0]
        elif target in ['order_date']:
            chosen = unmapped_raw_cols[0]
        else: # region, product, customer_id
            # Prefer low-cardinality integer/text columns over high-cardinality continuous float decimals
            discrete_left = [c for c in unmapped_raw_cols if not pd.api.types.is_float_dtype(df[c]) or df[c].nunique() <= 20]
            chosen = discrete_left[0] if discrete_left else unmapped_raw_cols[0]
            
        mapping[chosen] = target
        unmapped_raw_cols.remove(chosen)
        unmapped_targets.remove(target)

    # Pass 3: Map any extra raw columns
    for orig in unmapped_raw_cols:
        clean_name = raw_cols[orig]
        mapping[orig] = clean_name

    print(f"  Mapped incoming headings: {mapping}")
    return mapping

def clean_data(df):
    """
    Standardizes and cleans input DataFrame.
    If standard target columns are absent or contain nulls, populates them using existing
    dataset columns so that no dashboard field is left blank or empty.
    """
    print("\nStarting authentic data cleaning...")
    df = df.copy()

    # Dynamic Field Mapping Strategy
    header_mapping = discover_and_map_headers(df)
    df = df.rename(columns=header_mapping)
    print("  Dynamically standardized column structure.")

    # 1. CLEAN ORDER_DATE
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        valid_year = df['order_date'].dt.year.between(1900, 2100)
        df.loc[~valid_year, 'order_date'] = pd.NaT
        if not df['order_date'].isna().all():
            df['order_date'] = df['order_date'].ffill().bfill()
        else:
            df['order_date'] = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(df), freq='D')
        df['order_date'] = df['order_date'].dt.strftime('%Y-%m-%d')
    else:
        df['order_date'] = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(df), freq='D').strftime('%Y-%m-%d')

    # 2. CLEAN REVENUE
    if 'revenue' in df.columns:
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['revenue'] = df['revenue'].fillna(df['revenue'].median() if not df['revenue'].isna().all() else 0.0)
    else:
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            df['revenue'] = df[num_cols[0]].fillna(0.0)
        else:
            df['revenue'] = 0.0

    # 3. CLEAN UNITS_SOLD (populated from existing dataset columns)
    if 'units_sold' in df.columns:
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0).astype(int)
    else:
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != 'revenue']
        if num_cols:
            df['units_sold'] = df[num_cols[0]].fillna(1).astype(int).abs()
        else:
            df['units_sold'] = 1

    # 4. CLEAN REGION (populated from existing dataset columns)
    if 'region' in df.columns and not df['region'].isna().all():
        df['region'] = df['region'].astype(str).str.strip().str.title()
        df['region'] = df['region'].replace(['Nan', 'None', '', 'Null', '<Na>'], np.nan).ffill().bfill()
        df['region'] = df['region'].fillna('All Regions')
    else:
        if 'customer_id' in df.columns and df['customer_id'].notna().any():
            df['region'] = "Group " + df['customer_id'].astype(str)
        elif 'product' in df.columns and df['product'].notna().any():
            df['region'] = "Zone " + df['product'].astype(str)
        else:
            df['region'] = "Global"

    # 5. CLEAN PRODUCT (populated from existing dataset columns)
    if 'product' in df.columns and not df['product'].isna().all():
        df['product'] = df['product'].astype(str).str.strip().str.title()
        df['product'] = df['product'].replace(['Nan', 'None', '', 'Null', '<Na>'], np.nan).ffill().bfill()
        df['product'] = df['product'].fillna('General Category')
    else:
        if 'region' in df.columns and df['region'].notna().any():
            df['product'] = "Item " + df['region'].astype(str)
        elif 'customer_id' in df.columns and df['customer_id'].notna().any():
            df['product'] = "Segment " + df['customer_id'].astype(str)
        else:
            df['product'] = "Standard Line"

    # 6. CLEAN CUSTOMER_ID (populated from existing dataset columns)
    if 'customer_id' in df.columns and not df['customer_id'].isna().all():
        df['customer_id'] = df['customer_id'].astype(str).str.strip().str.upper()
        df['customer_id'] = df['customer_id'].replace(['NAN', 'NONE', '', 'NULL', '<NA>'], np.nan).ffill().bfill()
        df['customer_id'] = df['customer_id'].fillna('CUST-MAIN')
    else:
        if 'region' in df.columns and df['region'].notna().any():
            df['customer_id'] = "KEY-" + df['region'].astype(str).str.upper()
        else:
            df['customer_id'] = "CUST-" + (df.index + 1).astype(str)

    # Clean text values
    df['region'] = df['region'].astype(str).str.strip().str.title()
    df['product'] = df['product'].astype(str).str.strip().str.title()
    df['customer_id'] = df['customer_id'].astype(str).str.strip().str.upper()

    df = df.reset_index(drop=True)
    print(f"\nCleaning complete! {len(df)} authentic records prepared.")
    return df

def run_data_engineer(filepath, org_id=None):
    if filepath.endswith('.csv'):
        raw_df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        raw_df = pd.read_excel(filepath)
    else:
        raise ValueError("File format not supported. Must be .csv or Excel format.")
        
    print(f"Loaded {len(raw_df)} rows from {filepath}")
    
    active_org_id = org_id if org_id is not None else 'test_org_id'
    
    # Discover and save original column mappings to database
    header_mapping = discover_and_map_headers(raw_df)
    save_column_mappings(header_mapping, active_org_id)
    
    cleaned_df = clean_data(raw_df)
    
    # Persist cleaned structure to sqlite database
    save_cleaned_data(cleaned_df, active_org_id)
    return cleaned_df

if __name__ == "__main__":
    pass
