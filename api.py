import os
import pandas as pd
from datetime import datetime
from flask import Blueprint, request, jsonify, session, current_app
from werkzeug.utils import secure_filename

# Import agent runner functions
from data_engineer_agent import run_data_engineer
from analyst_agent import run_analyst
from insight_agent import run_insight_agent

# Import database methods
from database import get_table_as_df

# Import shared utilities
from utils import login_required, allowed_file, safe_json, get_kpi_data

# Import chart integrations
from charts import get_chart_response, _get_period_freq

api_bp = Blueprint('api', __name__)

@api_bp.route("/upload", methods=["POST"])
@login_required
def api_upload():
    """API endpoint for file uploads from static HTML pages."""
    org_id = session['org_id']

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected!'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create organization isolated uploads folder
        org_upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], org_id)
        os.makedirs(org_upload_dir, exist_ok=True)
        filepath = os.path.join(org_upload_dir, filename)
        file.save(filepath)

        # Store in session so it remains isolated per organization
        session['dataset_path'] = filepath
        session['dataset_name'] = filename

        return jsonify({
            'status': 'success', 
            'message': f'Successfully uploaded {filename}!',
            'filename': filename
        })
    else:
        return jsonify({
            'status': 'error', 
            'message': 'Invalid file type. Please upload .csv, .xlsx, or .xls'
        }), 400

@api_bp.route("/save-pdf", methods=["POST"])
@login_required
def api_save_pdf():
    """API endpoint to save the generated PDF file on the server's filesystem."""
    if 'pdf' not in request.files:
        return jsonify({'status': 'error', 'message': 'No PDF data received'}), 400
    
    pdf_file = request.files['pdf']
    org_id = session['org_id']
    
    # Save in organization-isolated uploads directory
    org_upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], org_id)
    pdf_exports_dir = os.path.join(org_upload_dir, 'pdf_reports')
    os.makedirs(pdf_exports_dir, exist_ok=True)
    
    # Make a clean filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"Sales_Performance_Dashboard_{timestamp}.pdf"
    filepath = os.path.join(pdf_exports_dir, filename)
    
    pdf_file.save(filepath)
    print(f"\n[PDF Export] Saved dashboard PDF to server filesystem: {filepath}")
    
    return jsonify({
        'status': 'success',
        'message': 'PDF saved successfully in the server filesystem.',
        'filename': filename,
        'filepath': filepath
    })

@api_bp.route("/run-analysis", methods=["POST"])
@login_required
def run_analysis():
    """Trigger the full agent pipeline via API."""
    org_id = session['org_id']
    try:
        # Require an uploaded file to run analysis
        dataset_path = session.get('dataset_path')
        dataset_name = session.get('dataset_name')
        
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': 'No dataset has been uploaded yet. Please go to the Upload page and submit your data before running analysis.'
            }), 400

        data_path = dataset_path
        print(f"\nUsing uploaded dataset: {dataset_name} for Org {org_id}")

        print(f"\nStarting full analysis pipeline for Org {org_id}...")
        cleaned_df = run_data_engineer(data_path, org_id)
        analysis_results = run_analyst(org_id)
        insights = run_insight_agent(analysis_results, org_id)

        return jsonify({
            'status': 'success',
            'message': 'Analysis complete!',
            'dataset': dataset_name if (dataset_path and os.path.exists(dataset_path)) else 'sample_sales.csv',
            'records_processed': len(cleaned_df),
            'insights_generated': len(insights),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route("/charts/<chart_name>.png")
@login_required
def api_chart_image(chart_name):
    """API route for serving dynamic PNG charts."""
    org_id = session['org_id']
    return get_chart_response(chart_name, org_id)

@api_bp.route("/<data_type>")
@login_required
def api_get_tenant_data(data_type):
    """
    Consolidated dynamic API endpoint to retrieve organization-isolated data.
    Using dynamic routing (<data_type>) removes code duplication and makes it beginner-friendly.
    """
    table_mapping = {
        'insights': 'insights',
        'sales': 'sales_clean',
        'forecasts': 'forecasts',
        'anomalies': 'anomalies',
        'segments': 'customer_segments'
    }
    
    table_name = table_mapping.get(data_type.lower())
    if not table_name:
        return jsonify({'status': 'error', 'message': f"Endpoint '/api/{data_type}' not found"}), 404
        
    org_id = session['org_id']
    df = get_table_as_df(table_name, org_id)
    return jsonify({'status': 'success', 'count': len(df), 'data': safe_json(df)})

@api_bp.route("/kpis")
@login_required
def api_kpis():
    org_id = session['org_id']
    return jsonify({'status': 'success', 'data': get_kpi_data(org_id)})

@api_bp.route("/revenue-by-region")
@login_required
def api_revenue_by_region():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    grouped = df.groupby('region')['revenue'].sum().reset_index()
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@api_bp.route("/revenue-by-product")
@login_required
def api_revenue_by_product():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0 or 'product' not in df.columns: return jsonify({'status': 'success', 'data': []})
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    grouped = df.groupby('product')['revenue'].sum().reset_index()
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@api_bp.route("/units-by-region")
@login_required
def api_units_by_region():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0 or 'region' not in df.columns or 'units_sold' not in df.columns: return jsonify({'status': 'success', 'data': []})
    grouped = df.groupby('region')['units_sold'].sum().reset_index()
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@api_bp.route("/revenue-trend")
@login_required
def api_revenue_trend():
    org_id = session['org_id']
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df.dropna(subset=['order_date'])
    if df.empty: return jsonify({'status': 'success', 'data': []})
    freq, label = _get_period_freq(df)
    df['period'] = df['order_date'].dt.to_period(freq).astype(str)
    grouped = df.groupby('period')['revenue'].sum().reset_index()
    # rename period back to quarter so that the frontend parsing remains compatible
    grouped = grouped.rename(columns={'period': 'quarter'})
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@api_bp.route("/segment-distribution")
@login_required
def api_segment_distribution():
    org_id = session['org_id']
    df = get_table_as_df('customer_segments', org_id)
    if len(df) == 0: return jsonify({'status': 'success', 'data': []})
    grouped = df.groupby('segment').size().reset_index(name='count')
    return jsonify({'status': 'success', 'data': safe_json(grouped)})

@api_bp.route("/column-mappings")
@login_required
def api_column_mappings():
    org_id = session['org_id']
    from database import get_column_mappings
    mappings = get_column_mappings(org_id)
    # Default fallbacks
    default_mappings = {
        'order_date': 'Date',
        'region': 'Region',
        'product': 'Product',
        'revenue': 'Revenue',
        'units_sold': 'Units',
        'customer_id': 'Customer'
    }
    # Ensure every standard column has a title
    result_mappings = {}
    for k, default_val in default_mappings.items():
        val = mappings.get(k)
        if val is None or val == '' or val.startswith('category_'):
            result_mappings[k] = default_val
        else:
            result_mappings[k] = val.replace('_', ' ').strip().title()
    return jsonify({'status': 'success', 'data': result_mappings})

@api_bp.route("/dataset-info")
@login_required
def api_dataset_info():
    dataset_name = session.get('dataset_name', 'None')
    return jsonify({'status': 'success', 'dataset': dataset_name})

@api_bp.route("/health")
def health_check():
    """Health check endpoint for deployment platforms."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
