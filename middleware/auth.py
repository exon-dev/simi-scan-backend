# auth.py
import jwt
from flask import request, jsonify
from jwt import InvalidTokenError 
import os

SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET')

def is_authenticated():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False, jsonify({'Error': 'Authorization header is missing'}), 401

    try:
        token = auth_header.split(" ")[1]
        decoded_token = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
        return True, decoded_token, None
    except InvalidTokenError:
        return False, jsonify({'Error': 'Invalid or expired token'}), 401
    except Exception as e:
        return False, jsonify({'Error': str(e)}), 401
