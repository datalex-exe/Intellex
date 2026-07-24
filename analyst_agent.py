"""
analyst_agent.py - Dynamic Authentic Analysis Agent
Performs statistical analysis, anomaly detection, forecasting, and segmentation
strictly on authentic dataset metrics with standard statistical calculations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from database import get_table_as_df, save_anomalies, save_forecasts, save_segments

def schema_discovery(df):
    """Dynamically finds the best columns for analysis based on types and headings."""
    print("\nDiscovering schema dynamically...")
    columns = {
        'date': None,
        'numeric_target': None,  # Main target like revenue/sales
        'numeric_secondary': [], # Extra numeric targets like units_sold
        'categories': [],        # Grouping categories like region/product
        'id': None               # Unique identifier like customer_id
    }
    
    # 1. Detect Date Column
    for col in df.columns:
        if col == 'org_id':
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
            if df[col].notna().any():
                columns['date'] = col
                break
            
    # 2. Detect Numeric Targets (Revenue, Sales, Quantities)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'org_id']
    target_keywords = ['revenue', 'sales', 'amount', 'spend', 'total']
    for col in num_cols:
        if any(kw in col.lower() for kw in target_keywords) and df[col].notna().any():
            columns['numeric_target'] = col
            break
    if not columns['numeric_target'] and num_cols:
        for col in num_cols:
            if df[col].notna().any():
                columns['numeric_target'] = col
                break
        
    for col in num_cols:
        if col != columns['numeric_target'] and df[col].notna().any():
            columns['numeric_secondary'].append(col)

    # 3. Detect Categories and IDs (ignoring org_id)
    cat_cols = [c for c in df.columns if c not in num_cols and c != 'org_id' and c != columns['date']]
    for col in cat_cols:
        if not df[col].notna().any():
            continue
        clean_name = col.lower()
        if 'id' in clean_name or 'key' in clean_name or df[col].nunique() > len(df) * 0.6:
            if not columns['id']: 
                columns['id'] = col
        else:
            columns['categories'].append(col)
            
    if not columns['id'] and len(cat_cols) > 0:
        for col in cat_cols:
            if col not in columns['categories']:
                columns['id'] = col
                break
        
    print(f"  Detected Schema Rules: {columns}")
    return columns

def compute_summary_stats(df, schema):
    print("\nComputing dynamic summary statistics...")
    stats = {}
    target = schema['numeric_target']
    
    if not target or target not in df.columns or df[target].empty:
        print("  Notice: No valid numeric target found for summary calculation.")
        return stats

    # Nulls are stored as 0 — use all rows for the sum (0-filled rows contribute 0)
    valid_df = df.copy()
    # For average: only count rows with a real value (> 0) so zeros don't skew the avg
    nonzero_df = valid_df[valid_df[target] > 0]

    # Overall Summary
    stats['overall'] = {
        f'total_{target}': float(valid_df[target].sum()),
        f'avg_{target}': float(nonzero_df[target].mean()) if not nonzero_df.empty else 0.0,
        'total_records': len(valid_df)
    }
    if schema['id'] and schema['id'] in valid_df.columns:
        stats['overall'][f'unique_{schema["id"]}'] = int(valid_df[schema['id']].nunique())
    for sec in schema['numeric_secondary']:
        if sec in valid_df.columns:
            stats['overall'][f'total_{sec}'] = float(valid_df[sec].sum())

    # Dynamic Aggregation by available categorical dimensions
    for cat in schema['categories'][:2]:
        if cat in valid_df.columns and valid_df[cat].notna().any():
            agg_dict = {target: ['sum', 'mean', 'count']}
            for sec in schema['numeric_secondary']:
                if sec in valid_df.columns:
                    agg_dict[sec] = 'sum'
            grouped = valid_df.groupby(cat).agg(agg_dict).round(2)
            stats[f'by_{cat}'] = grouped
        
    # Periodic Trend if date exists
    if schema['date'] and schema['date'] in valid_df.columns:
        df_copy = valid_df.copy()
        df_copy['order_date_parsed'] = pd.to_datetime(df_copy[schema['date']], errors='coerce')
        valid_dates = df_copy.dropna(subset=['order_date_parsed'])
        if not valid_dates.empty:
            valid_dates['period'] = valid_dates['order_date_parsed'].dt.to_period('Q')
            stats['by_period'] = valid_dates.groupby('period')[target].agg(['sum', 'mean', 'count']).round(2)
        
    print("  Summary stats computed successfully.")
    return stats

def calculate_growth_rates(df, schema):
    if not schema['date'] or not schema['numeric_target'] or schema['date'] not in df.columns:
        return pd.DataFrame()
        
    target = schema['numeric_target']
    df_copy = df.copy()
    df_copy['parsed_date'] = pd.to_datetime(df_copy[schema['date']], errors='coerce')
    valid_df = df_copy.dropna(subset=['parsed_date', target])
    
    if len(valid_df) < 2:
        return pd.DataFrame()

    print("\nCalculating trend growth rates...")
    date_min = valid_df['parsed_date'].min()
    date_max = valid_df['parsed_date'].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
    
    if days_span <= 30:
        period_freq = 'D'
    elif days_span <= 180:
        period_freq = 'W'
    elif days_span <= 365:
        period_freq = 'M'
    else:
        period_freq = 'Q'
        
    valid_df['period'] = valid_df['parsed_date'].dt.to_period(period_freq)
    
    periodic = valid_df.groupby('period')[target].sum().reset_index()
    periodic['period_str'] = periodic['period'].astype(str)
    periodic['pop_growth'] = periodic[target].pct_change() * 100
    periodic['pop_growth'] = periodic['pop_growth'].round(2)
    print("  Growth rates calculated.")
    return periodic

def detect_anomalies(df, schema):
    target = schema['numeric_target']
    if not target or target not in df.columns:
        return pd.DataFrame()
        
    feature_cols = [target] + [c for c in schema['numeric_secondary'] if c in df.columns]
    valid_df = df.dropna(subset=[target]).copy()
    
    if len(valid_df) < 5:
        return pd.DataFrame()

    print("\nDetecting multi-variate anomalies...")
    features = valid_df[feature_cols].fillna(0).values
    
    iso_forest = IsolationForest(
        contamination=0.1, 
        random_state=42, 
        n_estimators=30, 
        max_samples=min(1000, len(features)), 
        n_jobs=1
    )
    predictions = iso_forest.fit_predict(features)
    scores = iso_forest.decision_function(features)
    valid_df['anomaly_score'] = scores
    valid_df['is_anomaly'] = predictions == -1

    valid_df['severity'] = valid_df['anomaly_score'].apply(lambda s: 'High' if s < -0.3 else ('Medium' if s < -0.15 else 'Low'))
    anomalies = valid_df[valid_df['is_anomaly']].copy()
    
    keep_cols = [c for c in ([schema['date']] if schema['date'] else []) + schema['categories'] + [target, 'anomaly_score', 'severity'] if c in valid_df.columns]
    print(f"  Found {len(anomalies)} anomalies based on variables {feature_cols}")
    return anomalies[keep_cols]

def forecast_trends(df, schema):
    target = schema['numeric_target']
    date_col = schema['date']
    if not date_col or not target or date_col not in df.columns or target not in df.columns:
        return pd.DataFrame()
        
    df_copy = df.copy()
    df_copy['parsed_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
    valid_df = df_copy.dropna(subset=['parsed_date', target])
    
    if len(valid_df) < 2:
        return pd.DataFrame()

    print("\nForecasting targeted vectors with authentic standard error bounds...")
    forecasts = []
    
    date_min = valid_df['parsed_date'].min()
    date_max = valid_df['parsed_date'].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
    if days_span <= 30:
        period_freq = 'D'
    elif days_span <= 180:
        period_freq = 'W'
    elif days_span <= 365:
        period_freq = 'M'
    else:
        period_freq = 'Q'
        
    valid_df['period'] = valid_df['parsed_date'].dt.to_period(period_freq)
    
    primary_cat = schema['categories'][0] if schema['categories'] and schema['categories'][0] in valid_df.columns else None
    
    if primary_cat and valid_df[primary_cat].notna().any():
        top_groups = valid_df.groupby(primary_cat)[target].sum().sort_values(ascending=False).head(10).index.tolist()
    else:
        top_groups = ['Overall']
        valid_df['Overall'] = 'Overall'
        primary_cat = 'Overall'
    
    for group in top_groups:
        group_df = valid_df[valid_df[primary_cat] == group]
        periodic = group_df.groupby('period')[target].sum().reset_index()
        periodic['period_num'] = range(len(periodic))
        
        if len(periodic) >= 2:
            x = periodic['period_num'].values
            y = periodic[target].values
            n = len(x)
            
            denom = (n * np.sum(x**2) - np.sum(x)**2)
            if denom == 0:
                m = 0
                b = np.mean(y)
            else:
                m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denom
                b = (np.sum(y) - m * np.sum(x)) / n
            
            next_x = len(x)
            predicted = max(0.0, m * next_x + b)
            
            # Calculate authentic Residual Standard Error (S_e)
            y_pred = m * x + b
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            
            df_freedom = n - 2 if n > 2 else 1
            s_e = np.sqrt(ss_res / df_freedom)
            
            # Prediction interval margin of error (approx 95% confidence using 1.96 * S_e)
            mean_x = np.mean(x)
            sum_sq_x = np.sum((x - mean_x)**2)
            
            if sum_sq_x > 0:
                leverage = 1.0 + (1.0 / n) + ((next_x - mean_x)**2 / sum_sq_x)
            else:
                leverage = 1.0 + (1.0 / n)
                
            margin_of_error = 1.96 * s_e * np.sqrt(leverage)
            
            # Fallback if margin of error is zero (e.g. perfectly flat line or only 2 points)
            if margin_of_error == 0:
                margin_of_error = 0.10 * predicted if predicted > 0 else 10.0
                
            lower_bound = max(0.0, predicted - margin_of_error)
            upper_bound = predicted + margin_of_error
            
            last_period = periodic['period'].iloc[-1]
            next_period = last_period + 1
            
            forecasts.append({
                'region': str(group),
                'product': None,
                'forecast_date': str(next_period.end_time.date()),
                'predicted_revenue': round(float(predicted), 2),
                'lower_bound': round(float(lower_bound), 2),
                'upper_bound': round(float(upper_bound), 2)
            })
            
    print(f"  Generated {len(forecasts)} forecasts matching dimension: {primary_cat}")
    return pd.DataFrame(forecasts)

def segment_entities(df, schema):
    target = schema['numeric_target']
    entity_id = schema['id']
    if not target or not entity_id or target not in df.columns or entity_id not in df.columns:
        return pd.DataFrame()
        
    valid_df = df.dropna(subset=[target, entity_id])
    if len(valid_df) == 0:
        return pd.DataFrame()

    print(f"\nSegmenting core entities based on {entity_id}...")
    entity_spend = valid_df.groupby(entity_id)[target].sum().reset_index()
    entity_spend.columns = ['customer_id', 'total_spend']
    
    if len(entity_spend) < 2:
        entity_spend['segment'] = 'Single Segment'
        return entity_spend

    q25 = entity_spend['total_spend'].quantile(0.25)
    q75 = entity_spend['total_spend'].quantile(0.75)

    def assign_segment(spend):
        if spend >= q75: return 'High Value'
        elif spend >= q25: return 'Mid Value'
        else: return 'Low Value'

    entity_spend['segment'] = entity_spend['total_spend'].apply(assign_segment)
    print(f"  Segmented {len(entity_spend)} entities via distribution analysis.")
    return entity_spend

def run_analyst(org_id=None):
    print("\n" + "="*50)
    print("DYNAMIC ANALYST AGENT STARTING")
    print("="*50)
    
    active_org_id = org_id if org_id is not None else 'test_org_id'
    
    df = get_table_as_df('sales_clean', active_org_id)
    if len(df) == 0:
        print(f"No data found for Org {active_org_id}! Run Data Engineer first.")
        return None
        
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
    # Discover headings and types dynamically
    schema = schema_discovery(df)
    
    results = {
        'summary_stats': compute_summary_stats(df, schema),
        'growth_rates': calculate_growth_rates(df, schema),
        'anomalies': detect_anomalies(df, schema),
        'forecasts': forecast_trends(df, schema),
        'segments': segment_entities(df, schema)
    }
    
    # Save results to base schema layout for API extraction
    save_anomalies(results['anomalies'], active_org_id)
    save_forecasts(results['forecasts'], active_org_id)
    save_segments(results['segments'], active_org_id)
    
    print("\n" + "="*50)
    print("DYNAMIC ANALYST AGENT COMPLETE")
    print("="*50)
    return results

if __name__ == "__main__":
    results = run_analyst()
