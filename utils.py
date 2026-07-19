from functools import wraps
import math
import pandas as pd
from flask import session, request, jsonify, redirect, url_for

# Import configuration and database helpers
from config import ALLOWED_EXTENSIONS
from database import get_table_as_df

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # For API endpoints, return JSON
            if request.path.startswith('/api/'):
                return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
            # For pages, redirect to login
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_json(data):
    if isinstance(data, pd.DataFrame):
        return safe_json(data.astype(object).where(pd.notna(data), None).to_dict(orient='records'))
    if isinstance(data, pd.Series):
        return safe_json(data.astype(object).where(pd.notna(data), None).to_dict())
    if isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [safe_json(v) for v in data]
    if isinstance(data, tuple):
        return [safe_json(v) for v in data]
    if isinstance(data, (pd.Timestamp, pd.Timedelta)):
        return str(data)
    if pd.isna(data):
        return None
    if isinstance(data, (float, int)) and (math.isnan(float(data)) or math.isinf(float(data))):
        return None
    return data

def get_kpi_data(org_id):
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0:
        return {'total_revenue': 0, 'total_orders': 0, 'total_units': 0,
                'active_customers': 0, 'avg_order_value': 0}
    return {
        'total_revenue': float(df['revenue'].sum()),
        'total_orders': len(df),
        'total_units': int(df['units_sold'].sum()),
        'active_customers': int(df['customer_id'].nunique()),
        'avg_order_value': float(df['revenue'].mean())
    }

def create_sample_data(filepath):
    """Create sample sales data for demo purposes."""
    import numpy as np
    np.random.seed(42)
    
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    data = []
    for i in range(500):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        region = np.random.choice(regions)
        product = np.random.choice(products)
        revenue = np.random.uniform(1000, 50000)
        units = np.random.randint(1, 500)
        customer_id = f'CUST{np.random.randint(1000, 9999):04d}'
        
        data.append({
            'order_date': date.strftime('%Y-%m-%d'),
            'region': region,
            'product': product,
            'revenue': round(revenue, 2),
            'units_sold': units,
            'customer_id': customer_id
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Created sample data at {filepath}")

