"""
app.py - Production Flask Web Application (Refactored)
=====================================================
Features:
- Orchestrator of the application routes
- Registers modular auth and api Blueprints
- Serves static pages/routes (landing page, login, register, upload, dashboard)
"""

import os
from flask import Flask, session, redirect, url_for, send_from_directory
from flask_cors import CORS

# Import centralized configuration settings
from config import SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH

# Import our database operations
from database import init_database

# Import Blueprints
from auth import auth_bp
from api import api_bp

# Import shared utilities
from utils import login_required

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = SECRET_KEY
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Enable CORS for all domains (configure for production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# Upload config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_database()

# Register Blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(api_bp, url_prefix='/api')

# ============================================================
# PAGES (HTML serving)
# ============================================================

@app.route("/")
def home():
    """Landing page - serves index.html."""
    return send_from_directory('static', 'index.html')

@app.route("/login")
def login_page():
    """Serving organization login page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return send_from_directory('static', 'login.html')

@app.route("/register")
def register_page():
    """Serving organization registration page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return send_from_directory('static', 'register.html')

@app.route("/upload")
@login_required
def upload_page():
    """Upload page - requires login."""
    return send_from_directory('static', 'upload.html')

@app.route("/dashboard")
@login_required
def dashboard():
    """Dashboard page - requires login."""
    return send_from_directory('static', 'dashboard.html')

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\nStarting Flask server on port {port}...")
    print(f"Home:       http://localhost:{port}/")
    print(f"Login:      http://localhost:{port}/login")
    print(f"Register:   http://localhost:{port}/register")
    print(f"Upload:     http://localhost:{port}/upload")
    print(f"Dashboard:  http://localhost:{port}/dashboard")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
