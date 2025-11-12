"""
Flask API for Ollama Training Service
✅ FIXED: datetime.UTC → datetime.utcnow() for Python 3.10 compatibility
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import threading
import subprocess
import json
import os
from datetime import datetime
import time

app = Flask(__name__)

# ✅ CRITICAL: Enable CORS with explicit configuration
CORS(app, 
     origins="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     expose_headers=["Content-Type"],
     supports_credentials=False,
     max_age=3600)

# In-memory training status store
training_jobs = {}
training_lock = threading.Lock()

APP_VERSION = "2.3-DATETIME-FIX"

# ✅ Add OPTIONS handler for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ✅ Add request logging
@app.before_request
def log_request():
    print(f"📥 {request.method} {request.path} from {request.remote_addr}")
    if request.method in ['POST', 'PUT']:
        print(f"   Content-Type: {request.content_type}")
        print(f"   Content-Length: {request.content_length}")
    return None

def run_training_script(training_id, params):
    """
    Run training script in background thread
    Updates training_jobs dict with status
    """
    try:
        training_jobs[training_id]['status'] = 'running'
        training_jobs[training_id]['started_at'] = datetime.utcnow().isoformat()
        
        # Prepare command
        cmd = [
            '/opt/venv/bin/python3',
            '/scripts/train_lora.py',
            params['postgres_uri'],
            params['user_id'],
            params['base_model'],
            params['adapter_version']
        ]
        
        print(f"🚀 Starting training: {' '.join(cmd)}")
        
        # Run training script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                'OUTPUT_DIR': params['output_dir'],
            }
        )
        
        training_jobs[training_id]['pid'] = process.pid
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        # Parse metadata from output
        metadata = None
        if '===METADATA_START===' in stdout:
            start = stdout.find('===METADATA_START===') + len('===METADATA_START===')
            end = stdout.find('===METADATA_END===')
            metadata_json = stdout[start:end].strip()
            try:
                metadata = json.loads(metadata_json)
            except:
                pass
        
        if process.returncode == 0:
            training_jobs[training_id]['status'] = 'completed'
            training_jobs[training_id]['completed_at'] = datetime.utcnow().isoformat()
            training_jobs[training_id]['metadata'] = metadata
            training_jobs[training_id]['final_loss'] = metadata.get('metrics', {}).get('loss') if metadata else None
            print(f"✅ Training {training_id} completed successfully")
        else:
            training_jobs[training_id]['status'] = 'failed'
            training_jobs[training_id]['completed_at'] = datetime.utcnow().isoformat()
            training_jobs[training_id]['error'] = stderr or 'Training script failed'
            print(f"❌ Training {training_id} failed: {stderr}")
        
        # Store logs
        training_jobs[training_id]['stdout'] = stdout
        training_jobs[training_id]['stderr'] = stderr
        
    except Exception as e:
        training_jobs[training_id]['status'] = 'failed'
        training_jobs[training_id]['error'] = str(e)
        training_jobs[training_id]['completed_at'] = datetime.utcnow().isoformat()
        print(f"❌ Training {training_id} exception: {e}")

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'ollama-training',
        'version': APP_VERSION,
        'status': 'running',
        'endpoints': ['/health', '/api/train', '/api/train/status/<id>', '/api/train/list']
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ollama-training',
        'version': APP_VERSION,
        'timestamp': datetime.utcnow().isoformat(),
        'active_trainings': len([j for j in training_jobs.values() if j['status'] == 'running'])
    })

@app.route('/api/train', methods=['GET', 'POST', 'OPTIONS'])
def train_endpoint():
    """
    Training endpoint - supports both GET (for debug) and POST
    
    POST Request body:
    {
        "user_id": "user-xxx",
        "adapter_version": "v1731234567890",
        "training_id": "train-xxx-v1731234567890",
        "postgres_uri": "postgresql://...",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_dir": "/models/adapters/user-xxx/v1731234567890"
    }
    """
    
    # ✅ Handle OPTIONS request (CORS preflight)
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    # ✅ Handle GET request (for debugging - return endpoint info)
    if request.method == 'GET':
        return jsonify({
            'endpoint': '/api/train',
            'method': 'POST',
            'description': 'Start LoRA training job',
            'required_fields': ['user_id', 'adapter_version', 'training_id', 'postgres_uri', 'base_model'],
            'optional_fields': ['output_dir'],
            'current_active_jobs': len([j for j in training_jobs.values() if j['status'] == 'running']),
            'total_jobs': len(training_jobs),
            'note': 'This is GET response - use POST to start training'
        }), 200
    
    # ✅ Handle POST request (actual training)
    print(f"📥 Received POST request to /api/train")
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
            
        print(f"📦 Request data: {list(data.keys())}")
        
        # Validate required fields
        required = ['user_id', 'adapter_version', 'training_id', 'postgres_uri', 'base_model']
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}'
            }), 400
        
        training_id = data['training_id']
        
        # Check if training already exists
        if training_id in training_jobs:
            existing = training_jobs[training_id]
            if existing['status'] == 'running':
                return jsonify({
                    'success': False,
                    'error': 'Training already in progress',
                    'training_id': training_id,
                    'status': existing['status']
                }), 409
        
        # Check concurrent training limit
        with training_lock:
            running_count = len([j for j in training_jobs.values() if j['status'] == 'running'])
            if running_count >= 1:  # Limit: 1 concurrent training
                return jsonify({
                    'success': False,
                    'error': 'Another training is already in progress. Please wait.'
                }), 429
            
            # Initialize training job
            training_jobs[training_id] = {
                'training_id': training_id,
                'user_id': data['user_id'],
                'adapter_version': data['adapter_version'],
                'base_model': data['base_model'],
                'status': 'initializing',
                'created_at': datetime.utcnow().isoformat(),
                'progress': 0
            }
        
        # Set output directory
        output_dir = data.get('output_dir', f"/models/adapters/{data['user_id']}/{data['adapter_version']}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Start training in background thread
        params = {
            'postgres_uri': data['postgres_uri'],
            'user_id': data['user_id'],
            'base_model': data['base_model'],
            'adapter_version': data['adapter_version'],
            'output_dir': output_dir
        }
        
        thread = threading.Thread(
            target=run_training_script,
            args=(training_id, params),
            daemon=True
        )
        thread.start()
        
        print(f"✅ Training {training_id} started in background")
        
        return jsonify({
            'success': True,
            'training_id': training_id,
            'status': 'running',
            'message': 'Training started successfully',
            'estimated_time': '10-30 minutes'
        })
        
    except Exception as e:
        print(f"❌ Start training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/train/status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """Get training status by ID"""
    try:
        if training_id not in training_jobs:
            return jsonify({
                'success': False,
                'error': 'Training not found'
            }), 404
        
        job = training_jobs[training_id]
        
        # Calculate progress (rough estimate)
        progress = 0
        if job['status'] == 'running':
            # Estimate based on time elapsed (assume 20 min average)
            if 'started_at' in job:
                elapsed = (datetime.utcnow() - datetime.fromisoformat(job['started_at'])).total_seconds()
                progress = min(int((elapsed / (20 * 60)) * 100), 95)
        elif job['status'] == 'completed':
            progress = 100
        
        response = {
            'success': True,
            'training_id': training_id,
            'status': job['status'],
            'progress': progress,
            'user_id': job.get('user_id'),
            'adapter_version': job.get('adapter_version'),
            'created_at': job.get('created_at'),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
        }
        
        # Add extra info based on status
        if job['status'] == 'completed':
            response['metadata'] = job.get('metadata')
            response['final_loss'] = job.get('final_loss')
        elif job['status'] == 'failed':
            response['error'] = job.get('error')
            # Include last 500 chars of stderr
            if 'stderr' in job:
                response['error_details'] = job['stderr'][-500:]
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Get status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train/list', methods=['GET'])
def list_trainings():
    """List all training jobs"""
    try:
        jobs_list = [
            {
                'training_id': tid,
                'user_id': job.get('user_id'),
                'status': job['status'],
                'created_at': job.get('created_at'),
            }
            for tid, job in training_jobs.items()
        ]
        
        return jsonify({
            'success': True,
            'total': len(jobs_list),
            'jobs': jobs_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import socket
    
    print("========================================")
    print("🚀 Starting Ollama Training API Server")
    print("========================================")
    print(f"📡 Version: {APP_VERSION}")
    print(f"📡 Host: 0.0.0.0")
    print(f"📡 Port: 5000")
    print(f"💻 Hostname: {socket.gethostname()}")
    print(f"📂 Working dir: {os.getcwd()}")
    print("========================================")
    print("")
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"  {methods:20s} {rule}")
    print("")
    print("========================================")
    print("🎯 Starting server...")
    print("========================================")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
        exit(1)