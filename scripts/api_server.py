#!/usr/bin/env python3
"""
LoRA Training API Server - SIMPLIFIED & DEBUGGABLE
"""

import os
import sys
import subprocess
import asyncio
from datetime import datetime
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("🚀 API Server Starting...")
logger.info("="*60)

try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from pydantic import BaseModel
    import psycopg2
    logger.info("✅ Imports successful")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")
    sys.exit(1)

API_PORT = int(os.environ.get('API_PORT', 8000))
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/workspace/adapters')
MODEL_NAME = os.environ.get('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

logger.info(f"📊 Config: PORT={API_PORT}, OUTPUT={OUTPUT_PATH}")

app = FastAPI(title="LoRA Training Service")

training_status = {}

class TrainingRequest(BaseModel):
    user_id: str
    adapter_version: str
    postgres_uri: str
    model_name: str = MODEL_NAME

async def run_training_job(training_id: str, user_id: str, adapter_version: str, 
                          postgres_uri: str, model_name: str):
    """Run training with full logging"""
    try:
        logger.info(f"🚀 Starting training: {training_id}")
        logger.info(f"   User: {user_id}")
        logger.info(f"   Version: {adapter_version}")
        logger.info(f"   Model: {model_name}")
        
        training_status[training_id] = {
            'status': 'running',
            'started_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        env = os.environ.copy()
        env.update({
            'POSTGRES_URI': postgres_uri,
            'USER_ID': user_id,
            'MODEL_NAME': model_name,
            'OUTPUT_PATH': OUTPUT_PATH,
        })
        
        logger.info(f"📝 Executing: python3 /workspace/scripts/train_complete.py")
        
        # ✅ Run with real-time output
        process = subprocess.Popen(
            ['python3', '/workspace/scripts/train_complete.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr to stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[TRAIN] {line}")
                output_lines.append(line)
        
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"✅ Training completed: {training_id}")
            training_status[training_id] = {
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            # Update DB
            try:
                conn = psycopg2.connect(postgres_uri)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_data_schema.training_jobs
                    SET status = 'completed', completed_at = NOW()
                    WHERE job_id = %s
                """, (training_id,))
                conn.commit()
                cursor.close()
                conn.close()
                logger.info("✅ DB updated")
            except Exception as db_error:
                logger.error(f"⚠️  DB update error: {db_error}")
        else:
            error_output = '\n'.join(output_lines[-50:])  # Last 50 lines
            logger.error(f"❌ Training failed with exit code {process.returncode}")
            logger.error(f"Last output:\n{error_output}")
            
            training_status[training_id] = {
                'status': 'failed',
                'error': f"Exit code {process.returncode}",
                'completed_at': datetime.utcnow().isoformat() + 'Z'
            }
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        logger.error(traceback.format_exc())
        
        training_status[training_id] = {
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat() + 'Z'
        }

@app.get("/")
async def root():
    return {"service": "lora-training", "status": "online"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "active_trainings": len([s for s in training_status.values() if s.get('status') == 'running'])
    }

@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        training_id = f"train-{request.user_id[:8]}-{request.adapter_version}"
        
        logger.info(f"📥 Training request: {training_id}")
        
        # Check if already running
        existing = training_status.get(training_id)
        if existing and existing.get('status') == 'running':
            raise HTTPException(409, f"Already running: {training_id}")
        
        # Queue training
        training_status[training_id] = {
            'status': 'pending',
            'queued_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        background_tasks.add_task(
            run_training_job,
            training_id=training_id,
            user_id=request.user_id,
            adapter_version=request.adapter_version,
            postgres_uri=request.postgres_uri,
            model_name=request.model_name
        )
        
        logger.info(f"✅ Training queued: {training_id}")
        
        return {
            "success": True,
            "training_id": training_id,
            "adapter_version": request.adapter_version,
            "message": "Training started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Start training error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

@app.get("/status/{training_id}")
async def get_status(training_id: str):
    status = training_status.get(training_id)
    if not status:
        raise HTTPException(404, f"Not found: {training_id}")
    return {"training_id": training_id, **status}

@app.get("/list")
async def list_trainings():
    return {"trainings": list(training_status.values())}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*60)
    logger.info(f"🌐 Starting server on 0.0.0.0:{API_PORT}")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")