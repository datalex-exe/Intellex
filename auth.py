import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from werkzeug.security import check_password_hash

# Import centralized configuration settings
from config import (
    GETOTP_API_KEY, GETOTP_TEMPLATE_ID, GETOTP_SENDER
)

# Import our database operations
from database import (
    get_organization_by_join_code, get_organization_by_id,
    create_organization, create_user, get_user_by_email, get_user_by_phone
)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/send-otp", methods=["POST"])
def api_send_otp():
    data = request.get_json() or request.form
    action = data.get('action')  # 'create' or 'join'
    email = data.get('email')
    phone = data.get('phone')
    name = data.get('name')

    if not email or not phone or not name or not action:
        return jsonify({'status': 'error', 'message': 'Please fill all required fields.'}), 400

    # Clean up phone and email
    email_clean = email.strip().lower()
    phone_clean = ''.join(c for c in phone.strip() if c.isdigit())
    if len(phone_clean) == 10:
        phone_clean = '91' + phone_clean

    # Uniqueness checks
    existing_user = get_user_by_email(email_clean)
    if existing_user:
        return jsonify({'status': 'error', 'message': 'An account with this email already exists.'}), 400

    existing_phone = get_user_by_phone(phone_clean)
    if existing_phone:
        return jsonify({'status': 'error', 'message': 'A user with this phone number already exists.'}), 400

    if action == 'create':
        org_name = data.get('org_name')
        if not org_name:
            return jsonify({'status': 'error', 'message': 'Organization name is required.'}), 400
    elif action == 'join':
        join_code = data.get('join_code')
        if not join_code:
            return jsonify({'status': 'error', 'message': 'Organization Join Code is required.'}), 400
        org_details = get_organization_by_join_code(join_code)
        if not org_details:
            return jsonify({'status': 'error', 'message': 'Invalid Organization Join Code. Please check and try again.'}), 400

    # Call GetOTP API to send code via SMS
    url = "https://api.otp.dev/v1/verifications"
    headers = {
        "X-OTP-Key": GETOTP_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "data": {
            "channel": "sms",
            "sender": GETOTP_SENDER,
            "phone": phone_clean,
            "template": GETOTP_TEMPLATE_ID,
            "code_length": 6
        }
    }

    print(f"\n[GetOTP Request] Sender: '{GETOTP_SENDER}' | Phone: '{phone_clean}' | Template: '{GETOTP_TEMPLATE_ID}'")

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            # Response check
            if response.status in (200, 201):
                return jsonify({'status': 'success', 'message': 'OTP sent successfully!'})
            else:
                raise Exception(f"Non-200 status: {response.status} - {res_data}")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            err_data = json.loads(err_body)
            err_msg = err_data.get('message', err_body)
        except Exception:
            err_msg = str(e)
            
        print("\n" + "*" * 60)
        print(f"GetOTP API HTTP Error: {err_msg}")
        print("*" * 60 + "\n")
        
        return jsonify({
            'status': 'error',
            'message': f"Failed to send OTP via SMS: {err_msg}"
        }), 400
    except Exception as e:
        print("\n" + "*" * 60)
        print(f"GetOTP Connection Failure: {str(e)}")
        print("*" * 60 + "\n")
        
        return jsonify({
            'status': 'error',
            'message': f"Could not connect to OTP service: {str(e)}"
        }), 500

@auth_bp.route("/register", methods=["POST"])
def api_register():
    data = request.get_json() or request.form
    action = data.get('action')  # 'create' or 'join'
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    phone = data.get('phone')
    otp = data.get('otp')
    
    if not email or not password or not name or not action or not phone or not otp:
        return jsonify({'status': 'error', 'message': 'Please fill all required fields, including phone number and OTP.'}), 400
        
    phone_clean = ''.join(c for c in phone.strip() if c.isdigit())
    if len(phone_clean) == 10:
        phone_clean = '91' + phone_clean

    email_clean = email.strip().lower()
    existing_user = get_user_by_email(email_clean)
    if existing_user:
        return jsonify({'status': 'error', 'message': 'An account with this email already exists.'}), 400
        
    existing_phone = get_user_by_phone(phone_clean)
    if existing_phone:
        return jsonify({'status': 'error', 'message': 'A user with this phone number already exists.'}), 400

    # Verify OTP code via GetOTP API
    try:
        encoded_phone = urllib.parse.quote(phone_clean)
        verify_url = f"https://api.otp.dev/v1/verifications?code={otp.strip()}&phone={encoded_phone}"
        req = urllib.request.Request(verify_url, headers={
            "X-OTP-Key": GETOTP_API_KEY,
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            if not res_data.get('data'):
                return jsonify({'status': 'error', 'message': 'Invalid or expired OTP. Please verify and try again.'}), 400
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            err_data = json.loads(err_body)
            err_msg = err_data.get('message', err_body)
        except Exception:
            err_msg = str(e)
        return jsonify({'status': 'error', 'message': f'OTP verification failed: {err_msg}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'OTP verification service error: {str(e)}'}), 500
        
    try:
        if action == 'create':
            org_name = data.get('org_name')
            if not org_name:
                return jsonify({'status': 'error', 'message': 'Organization name is required.'}), 400
            
            # Create organization and user
            org_id, join_code = create_organization(org_name)
            user_id = create_user(org_id, email_clean, password, name, phone_clean, role='admin')
            org_details = get_organization_by_id(org_id)
            
            # Log in the user
            session['user_id'] = user_id
            session['org_id'] = org_id
            session['user_name'] = name
            session['org_name'] = org_details['name']
            
            return jsonify({
                'status': 'success',
                'message': 'Organization and Administrator accounts created successfully!',
                'join_code': join_code
            })
            
        elif action == 'join':
            join_code = data.get('join_code')
            if not join_code:
                return jsonify({'status': 'error', 'message': 'Organization Join Code is required.'}), 400
                
            org_details = get_organization_by_join_code(join_code)
            if not org_details:
                return jsonify({'status': 'error', 'message': 'Invalid Organization Join Code. Please check and try again.'}), 400
                
            # Create user linked to organization
            user_id = create_user(org_details['id'], email_clean, password, name, phone_clean, role='user')
            
            # Log in the user
            session['user_id'] = user_id
            session['org_id'] = org_details['id']
            session['user_name'] = name
            session['org_name'] = org_details['name']
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully joined organization {org_details["name"]}!'
            })
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid registration type.'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Registration failed: {str(e)}'}), 500

@auth_bp.route("/login", methods=["POST"])
def api_login():
    data = request.get_json() or request.form
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Please enter email and password.'}), 400
        
    user = get_user_by_email(email)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'status': 'error', 'message': 'Invalid email or password.'}), 400
        
    org = get_organization_by_id(user['org_id'])
    
    # Establish session
    session['user_id'] = user['id']
    session['org_id'] = user['org_id']
    session['user_name'] = user['name']
    session['org_name'] = org['name'] if org else 'Isolated Workspace'
    
    return jsonify({
        'status': 'success',
        'message': 'Logged in successfully!',
        'user_name': user['name'],
        'org_name': session['org_name']
    })

@auth_bp.route("/logout", methods=["GET", "POST"])
def api_logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logged out successfully.'})

@auth_bp.route("/status", methods=["GET"])
def api_auth_status():
    if 'user_id' in session:
        # Check org join code
        org_details = get_organization_by_id(session['org_id'])
        join_code = org_details['join_code'] if org_details else 'UNKNOWN'
        return jsonify({
            'logged_in': True,
            'user_name': session.get('user_name'),
            'org_name': session.get('org_name'),
            'org_id': session.get('org_id'),
            'join_code': join_code
        })
    return jsonify({'logged_in': False})
