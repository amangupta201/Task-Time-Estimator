from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import numpy as np
import pickle
import json
from typing import List, Optional
import os
from pathlib import Path

# Import your neural network classes
from main import NeuralNetwork, TaskTimeDataGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Neural Network Task Time Estimator API",
    description="A from-scratch neural network for predicting task completion times",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class TaskInput(BaseModel):
    task_category: int
    complexity: int
    description_length: int
    hour_of_day: int
    energy_level: int

    @validator('task_category')
    def validate_task_category(cls, v):
        if not 0 <= v <= 6:
            raise ValueError('task_category must be between 0 and 6')
        return v

    @validator('complexity')
    def validate_complexity(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('complexity must be between 1 and 10')
        return v

    @validator('description_length')
    def validate_description_length(cls, v):
        if not 5 <= v <= 100:
            raise ValueError('description_length must be between 5 and 100')
        return v

    @validator('hour_of_day')
    def validate_hour_of_day(cls, v):
        if not 8 <= v <= 18:
            raise ValueError('hour_of_day must be between 8 and 18')
        return v

    @validator('energy_level')
    def validate_energy_level(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('energy_level must be between 1 and 10')
        return v


class PredictionResponse(BaseModel):
    predicted_time: float
    confidence_interval: Optional[tuple] = None
    task_category_name: str
    model_info: dict


class BatchTaskInput(BaseModel):
    tasks: List[TaskInput]


class ModelInfo(BaseModel):
    architecture: List[int]
    total_parameters: int
    training_accuracy: dict
    feature_names: List[str]


# Global variables for model and data generator
model = None
data_generator = None
model_info = {}


def load_model():
    """Load the trained neural network model"""
    global model, data_generator, model_info

    # Check if saved model exists
    if os.path.exists('trained_model.pkl') and os.path.exists('model_info.json'):
        try:
            # Load saved model
            with open('trained_model.pkl', 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            data_generator = model_data['data_generator']

            with open('model_info.json', 'r') as f:
                model_info = json.load(f)

            print("✅ Loaded saved model")

        except Exception as e:
            print(f"⚠️ Error loading saved model: {e}")
            train_new_model()
    else:
        # Train a new model
        train_new_model()


def train_new_model():
    """Train a new neural network model"""
    global model, data_generator, model_info

    print("🚀 Training new neural network model...")

    # Generate synthetic data
    data_generator = TaskTimeDataGenerator(user_profile="balanced")
    X, y = data_generator.generate_task_features(n_samples=1000)

    # Normalize data
    X_norm = data_generator.normalize_features(X)
    y_norm = data_generator.normalize_targets(y)

    # Split data
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))

    X_train = X_norm[indices[:train_size]]
    y_train = y_norm[indices[:train_size]]
    X_val = X_norm[indices[train_size:]]
    y_val = y_norm[indices[train_size:]]

    # Create and train model
    model = NeuralNetwork(
        input_size=5,
        hidden_layers=[32, 16],
        output_size=1,
        learning_rate=0.01
    )

    print("🏃 Training neural network...")
    train_losses, val_losses = model.train(
        X_train, y_train, X_val, y_val,
        epochs=1000, batch_size=64, verbose=False
    )

    # Calculate metrics
    val_pred_norm = model.predict(X_val)
    val_pred = data_generator.denormalize_targets(val_pred_norm)
    y_val_orig = y[indices[train_size:]]

    val_mae = np.mean(np.abs(y_val_orig - val_pred))
    val_mape = np.mean(np.abs((y_val_orig - val_pred) / y_val_orig)) * 100

    # Store model info
    model_info = {
        "architecture": [5, 32, 16, 1],
        "total_parameters": sum([w.size for w in model.weights]) + sum([b.size for b in model.biases]),
        "training_accuracy": {
            "validation_mae": float(val_mae),
            "validation_mape": float(val_mape),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1])
        },
        "feature_names": ["task_category", "complexity", "description_length", "hour_of_day", "energy_level"]
    }

    # Save model
    try:
        model_data = {
            'model': model,
            'data_generator': data_generator
        }

        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        print("💾 Model saved successfully")

    except Exception as e:
        print(f"⚠️ Error saving model: {e}")

    print(f"✅ Model training complete! MAE: {val_mae:.2f} minutes")


# Load model on startup
load_model()


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse('static/index.html')


# Mount static files (CSS, JS, images, etc.)
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_generator_loaded": data_generator is not None
    }


@app.get("/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(**model_info)


@app.post("/predict")
async def predict_task_time(task: TaskInput):
    """Predict completion time for a single task"""
    if not model or not data_generator:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to numpy array
        features = np.array([[
            task.task_category,
            task.complexity,
            task.description_length,
            task.hour_of_day,
            task.energy_level
        ]])

        # Normalize features
        features_norm = data_generator.normalize_features(features)

        # Make prediction
        prediction_norm = model.predict(features_norm)
        prediction = data_generator.denormalize_targets(prediction_norm)

        # Get task category name
        category_names = ["coding", "writing", "research", "meeting", "admin", "creative", "debugging"]
        task_category_name = category_names[task.task_category]

        # Calculate approximate confidence interval (±1 std dev based on validation error)
        mae = model_info["training_accuracy"]["validation_mae"]
        confidence_interval = (
            float(prediction[0, 0] - mae),
            float(prediction[0, 0] + mae)
        )

        return PredictionResponse(
            predicted_time=float(prediction[0, 0]),
            confidence_interval=confidence_interval,
            task_category_name=task_category_name,
            model_info={
                "mae": model_info["training_accuracy"]["validation_mae"],
                "mape": model_info["training_accuracy"]["validation_mape"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch_tasks(batch: BatchTaskInput):
    """Predict completion times for multiple tasks"""
    if not model or not data_generator:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(batch.tasks) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tasks per batch")

    try:
        predictions = []
        category_names = ["coding", "writing", "research", "meeting", "admin", "creative", "debugging"]

        for task in batch.tasks:
            # Convert input to numpy array
            features = np.array([[
                task.task_category,
                task.complexity,
                task.description_length,
                task.hour_of_day,
                task.energy_level
            ]])

            # Normalize and predict
            features_norm = data_generator.normalize_features(features)
            prediction_norm = model.predict(features_norm)
            prediction = data_generator.denormalize_targets(prediction_norm)

            predictions.append({
                "task": task.dict(),
                "predicted_time": float(prediction[0, 0]),
                "task_category_name": category_names[task.task_category]
            })

        return {
            "predictions": predictions,
            "model_info": {
                "mae": model_info["training_accuracy"]["validation_mae"],
                "mape": model_info["training_accuracy"]["validation_mape"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/model/retrain")
async def retrain_model():
    """Retrain the model with new data (in production, you might want authentication)"""
    try:
        train_new_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "new_accuracy": model_info["training_accuracy"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@app.get("/model/weights")
async def get_model_weights():
    """Get model architecture and weight information (for debugging/analysis)"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        weights_info = []
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            weights_info.append({
                f"layer_{i + 1}": {
                    "weight_shape": w.shape,
                    "bias_shape": b.shape,
                    "weight_mean": float(np.mean(w)),
                    "weight_std": float(np.std(w)),
                    "bias_mean": float(np.mean(b))
                }
            })

        return {
            "architecture": model.layers,
            "weights_info": weights_info,
            "total_parameters": sum([w.size for w in model.weights]) + sum([b.size for b in model.biases])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting weights info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📊 Neural Network Task Time Estimator API")
    print("🌐 Frontend will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    # Use PORT environment variable for deployment, fallback to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
