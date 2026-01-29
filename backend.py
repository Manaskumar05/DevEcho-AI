

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import string

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

SECRET_KEY = "supersecretkeyforjwt_supercomplex_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logging.basicConfig(
    filename='ml_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------
# GLOBALS
# ---------------------------------------------------

fake_users_db: Dict[str, dict] = {}
trained_models: Dict[str, dict] = {}  
training_queue: List[dict] = []  

# ---------------------------------------------------
# SCHEMAS
# ---------------------------------------------------

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str
    role: str = "user"  # admin or user

class Token(BaseModel):
    access_token: str
    token_type: str

class TrainRequest(BaseModel):
    model_name: str
    epochs: int
    lr: float
    batch_size: int
    model_type: str = "mlp"  # mlp, cnn, dt

class PredictRequest(BaseModel):
    model_name: str
    features: List[float]

class MetricsResponse(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    model_name: str

# ---------------------------------------------------
# AUTH FUNCTIONS
# ---------------------------------------------------

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user: return False
    if not verify_password(password, user["password"]): return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None: raise credentials_exception
    return user

# ---------------------------------------------------
# DATA LOADER FUNCTIONS
# ---------------------------------------------------

def get_dummy_dataset(samples=2000, features=20, classes=3, batch_size=64):
    X, y = make_classification(n_samples=samples, n_features=features, n_classes=classes)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# ---------------------------------------------------
# MODEL CLASSES
# ---------------------------------------------------

class MLPModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32*20, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DummyDecisionTree:
    # Simulate a simple decision tree classifier
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    def fit(self, X, y):
        self.rules = [random.randint(0, self.num_classes-1) for _ in range(len(X[0]))]

    def predict(self, X):
        return [random.choice(self.rules) for _ in range(len(X))]

# ---------------------------------------------------
# TRAINING FUNCTION (ASYNC)
# ---------------------------------------------------

async def train_model_async(model_name: str, model_type: str, epochs=5, lr=0.001, batch_size=64):
    logging.info(f"Starting training for model: {model_name}, type: {model_type}")
    loader = get_dummy_dataset(batch_size=batch_size)
    if model_type == "mlp":
        model = MLPModel()
    elif model_type == "cnn":
        model = CNNModel()
    elif model_type == "dt":
        model = DummyDecisionTree()
    else:
        raise HTTPException(status_code=400, detail="Unknown model type")

    metrics = []

    if model_type in ["mlp", "cnn"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0
            all_true = []
            all_pred = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                if model_type == "cnn":
                    X_batch = X_batch.unsqueeze(1)  # add channel dim
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_true.extend(y_batch.tolist())
                all_pred.extend(predicted.tolist())
            accuracy = sum([a==b for a,b in zip(all_true, all_pred)])/len(all_true)
            precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
            recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
            f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
            metrics.append({
                "epoch": epoch+1,
                "loss": epoch_loss/len(loader),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            logging.info(f"{model_name} epoch {epoch+1} done: loss={epoch_loss/len(loader):.4f}, acc={accuracy:.4f}")
            await asyncio.sleep(0)
    else:
        # Decision tree simulation
        X = np.random.rand(1000, 20)
        y = np.random.randint(0, 3, size=1000)
        model.fit(X, y)
        metrics = [{"epoch": i+1, "loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0} for i in range(epochs)]
        await asyncio.sleep(0)

    # Save model and metrics
    version = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    trained_models[model_name] = {"model": model, "metrics": metrics, "version": version, "type": model_type}
    logging.info(f"Training completed for {model_name}, version {version}")
    return metrics

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

async def predict_model(model_name: str, features: List[float]):
    model_entry = trained_models.get(model_name)
    if not model_entry:
        raise HTTPException(status_code=400, detail="Model not found")
    model_type = model_entry["type"]
    model = model_entry["model"]
    if model_type in ["mlp", "cnn"]:
        x = torch.tensor([features], dtype=torch.float32)
        if model_type == "cnn":
            x = x.unsqueeze(1)
        model.eval()
        with torch.no_grad():
            output = model(x)
            _, predicted = torch.max(output, 1)
        return int(predicted.item())
    else:
        # Decision tree simulation
        return random.randint(0, model_entry["model"].num_classes-1)

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI(title="Ultra Complex MIT-Level AI/ML Backend API")

@app.post("/signup", tags=["auth"])
async def signup(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    fake_users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "password": get_password_hash(user.password),
        "role": user.role
    }
    logging.info(f"New user signed up: {user.username}")
    return {"msg": "User created successfully"}

@app.post("/login", response_model=Token, tags=["auth"])
async def login(user: UserCreate):
    auth_user = authenticate_user(user.username, user.password)
    if not auth_user:
        logging.warning(f"Failed login attempt for {user.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    logging.info(f"User logged in: {user.username}")
    return {"access_token": token, "token_type": "bearer"}

@app.post("/train", tags=["ml"])
async def train(req: TrainRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    job_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    training_queue.append({"job_id": job_id, "status": "queued", "model_name": req.model_name, "user": current_user["username"]})
    background_tasks.add_task(train_model_async, req.model_name, req.model_type, req.epochs, req.lr, req.batch_size)
    logging.info(f"Training job queued: {req.model_name} by {current_user['username']}")
    return {"job_id": job_id, "status": "queued", "model_name": req.model_name}

@app.post("/predict", tags=["ml"])
async def predict(req: PredictRequest, current_user: dict = Depends(get_current_user)):
    prediction = await predict_model(req.model_name, req.features)
    logging.info(f"Prediction requested by {current_user['username']} for model {req.model_name}: {prediction}")
    return {"prediction": prediction}

@app.get("/metrics/{model_name}", response_model=List[MetricsResponse], tags=["ml"])
async def get_metrics(model_name: str, current_user: dict = Depends(get_current_user)):
    model_entry = trained_models.get(model_name)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found")
    logging.info(f"Metrics requested for model {model_name} by {current_user['username']}")
    return model_entry["metrics"]

# ---------------------------------------------------
# SIMULATION OF LARGE MODELS / LONG TRAINING
# ---------------------------------------------------

@app.get("/simulate_massive_training", tags=["ml"])
async def simulate_massive_training(background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    # simulate 3 different models at once
    for i in range(3):
        model_name = f"sim_model_{i}_{datetime.utcnow().strftime('%f')}"
        background_tasks.add_task(train_model_async, model_name, "mlp", 10, 0.001, 128)
    logging.info(f"Massive training simulation triggered by {current_user['username']}")
    return {"msg": "Massive training simulation started"}

# ---------------------------------------------------
# TO RUN:
# uvicorn main:app --reload
# ---------------------------------------------------
