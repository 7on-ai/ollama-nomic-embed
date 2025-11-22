#!/usr/bin/env python3
"""
LoRA Training Service API Server - FIXED VERSION
- Better startup logging
- Proper error handling
- Health check that actually works
"""

import os
import sys
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import traceback

# ===== CRITICAL: Configure logging FIRST =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log startup immediately
logger.info("="*60)
logger.info("🚀 API Server Starting...")
logger.info("="*60)

try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import psycopg2
    logger.info("✅ Imports successful")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# ===== Configuration =====
API_PORT = int(os.environ.get('API_PORT', 8000))
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/models/adapters')
POSTGRES_URI = os.environ.get('POSTGRES_URI', '')
MODEL_NAME = os.environ.get('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

logger.info(f"📊 Configuration:")
logger.info(f"   API_PORT: {API_PORT}")
logger.info(f"   OUTPUT_PATH: {OUTPUT_PATH}")
logger.info(f"   MODEL_NAME: {MODEL_NAME}")
logger.info(f"   POSTGRES_URI: {'[SET]' if POSTGRES_URI else '[NOT SET]'}")

# ===== FastAPI App =====
app = FastAPI(
    title="LoRA Training Service",
    description="On-demand LoRA fine-tuning service",
    version="1.0.0"
)

# ===== Models =====
class TrainingRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    adapter_version: str = Field(..., description="Adapter version (e.g., v1234567890)")
    postgres_uri: Optional[str] = Field(None, description="Postgres connection string")
    model_name: Optional[str] = Field(None, description="Base model name")

class TrainingResponse(BaseModel):
    success: bool
    message: str
    training_id: str
    adapter_version: str
    output_path: str
    estimated_time: str = "10-30 minutes"

class StatusResponse(BaseModel):
    training_id: str
    status: str
    adapter_version: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

# ===== In-Memory Training Status =====
training_status = {}

def update_training_status(training_id: str, status: dict):
    training_status[training_id] = {
        **status,
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }

def get_training_status(training_id: str) -> Optional[dict]:
    return training_status.get(training_id)

async def run_training_job(
    training_id: str,
    user_id: str,
    adapter_version: str,
    postgres_uri: str,
    model_name: str
):
    """Run training job in background"""
    try:
        logger.info(f"🚀 Starting training: {training_id}")
        
        update_training_status(training_id, {
            'training_id': training_id,
            'status': 'running',
            'adapter_version': adapter_version,
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat() + 'Z'
        })
        
        env = os.environ.copy()
        env.update({
            'POSTGRES_URI': postgres_uri,
            'USER_ID': user_id,
            'MODEL_NAME': model_name,
            'ADAPTER_VERSION': adapter_version,
            'OUTPUT_PATH': OUTPUT_PATH,
            'VOLUME_MOUNT': OUTPUT_PATH,
        })
        
        output_dir = os.path.join(OUTPUT_PATH, user_id, adapter_version)
        
        logger.info(f"📝 Running training script...")
        process = subprocess.Popen(
            ['python3', '/workspace/scripts/train_complete.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"✅ Training completed: {training_id}")
            
            update_training_status(training_id, {
                'training_id': training_id,
                'status': 'completed',
                'adapter_version': adapter_version,
                'user_id': user_id,
                'output_path': output_dir,
                'completed_at': datetime.utcnow().isoformat() + 'Z'
            })
            
            try:
                conn = psycopg2.connect(postgres_uri)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_data_schema.training_jobs
                    SET status = 'completed',
                        completed_at = NOW()
                    WHERE job_id = %s
                """, (training_id,))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as db_error:
                logger.error(f"❌ DB update error: {db_error}")
        else:
            error_msg = stderr or "Training failed"
            logger.error(f"❌ Training failed: {training_id} - {error_msg}")
            
            update_training_status(training_id, {
                'training_id': training_id,
                'status': 'failed',
                'adapter_version': adapter_version,
                'user_id': user_id,
                'error': error_msg,
                'completed_at': datetime.utcnow().isoformat() + 'Z'
            })
            
    except Exception as e:
        logger.error(f"❌ Training error: {training_id} - {str(e)}")
        logger.error(traceback.format_exc())
        
        update_training_status(training_id, {
            'training_id': training_id,
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat() + 'Z'
        })

# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "lora-training",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "lora-training",
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "output_path": OUTPUT_PATH,
        "model_name": MODEL_NAME,
        "active_trainings": len([s for s in training_status.values() if s.get('status') == 'running'])
    }

@app.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start a new training job"""
    try:
        if not request.user_id or not request.adapter_version:
            raise HTTPException(
                status_code=400,
                detail="user_id and adapter_version are required"
            )
        
        postgres_uri = request.postgres_uri or POSTGRES_URI
        if not postgres_uri:
            raise HTTPException(
                status_code=400,
                detail="postgres_uri is required (provide in request or env)"
            )
        
        model_name = request.model_name or MODEL_NAME
        training_id = f"train-{request.user_id[:8]}-{request.adapter_version}"
        
        existing = get_training_status(training_id)
        if existing and existing['status'] == 'running':
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {training_id}"
            )
        
        update_training_status(training_id, {
            'training_id': training_id,
            'status': 'pending',
            'adapter_version': request.adapter_version,
            'user_id': request.user_id,
            'created_at': datetime.utcnow().isoformat() + 'Z'
        })
        
        background_tasks.add_task(
            run_training_job,
            training_id=training_id,
            user_id=request.user_id,
            adapter_version=request.adapter_version,
            postgres_uri=postgres_uri,
            model_name=model_name
        )
        
        output_path = os.path.join(OUTPUT_PATH, request.user_id, request.adapter_version)
        
        logger.info(f"✅ Training queued: {training_id}")
        
        return TrainingResponse(
            success=True,
            message="Training started successfully",
            training_id=training_id,
            adapter_version=request.adapter_version,
            output_path=output_path,
            estimated_time="10-30 minutes"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Start training error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{training_id}", response_model=StatusResponse)
async def get_status(training_id: str):
    """Get training status"""
    status = get_training_status(training_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Training not found: {training_id}"
        )
    
    return StatusResponse(**status)

@app.get("/list")
async def list_trainings():
    """List all training jobs"""
    return {
        "trainings": list(training_status.values()),
        "count": len(training_status)
    }

@app.delete("/cancel/{training_id}")
async def cancel_training(training_id: str):
    """Cancel a running training job"""
    status = get_training_status(training_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Training not found: {training_id}"
        )
    
    if status['status'] != 'running':
        raise HTTPException(
            status_code=400,
            detail=f"Training is not running: {status['status']}"
        )
    
    update_training_status(training_id, {
        **status,
        'status': 'cancelled',
        'completed_at': datetime.utcnow().isoformat() + 'Z'
    })
    
    return {
        "success": True,
        "message": "Training cancelled",
        "training_id": training_id
    }

# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("="*60)
    logger.info("✅ FastAPI Application Started")
    logger.info("="*60)
    logger.info(f"📂 Output Path: {OUTPUT_PATH}")
    logger.info(f"📦 Model: {MODEL_NAME}")
    logger.info(f"🌐 API Port: {API_PORT}")
    logger.info("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("👋 FastAPI Application Shutting Down")

# ===== Main Entry Point =====

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*60)
    logger.info("🌐 Starting Uvicorn Server...")
    logger.info(f"   Host: 0.0.0.0")
    logger.info(f"   Port: {API_PORT}")
    logger.info("="*60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=API_PORT,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Uvicorn failed to start: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)