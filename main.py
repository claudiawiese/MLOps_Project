import pdb
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc
import subprocess

# Configuration de l'application FastAPI
app = FastAPI()

# Load model 
model_uri = "models:/KNN_Accident_Model/1"  # Replace with your model's URI in MLflow
model = mlflow.pyfunc.load_model(model_uri)

# Clés de sécurité
SECRET_KEY = "votre_clé_secrète"  # Remplacez par une clé secrète
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Dépendance OAuth2 pour récupérer les tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Gestion du hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simuler une base de données d'utilisateurs
fake_users_db = {
    "admin": {"username": "admin", "role": "admin", "hashed_password": pwd_context.hash("admin123")},
    "manager": {"username": "manager", "role": "manager", "hashed_password": pwd_context.hash("manager123")},
    "user": {"username": "user", "role": "user", "hashed_password": pwd_context.hash("user123")}
}

# Classe pour stocker les données de l'utilisateur
class User(BaseModel):
    username: str
    role: str

class UserInDB(User):
    hashed_password: str

# Vérification du mot de passe
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Obtenir un utilisateur de la "fake DB"
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

# Création d'un token JWT pour l'authentification
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Endpoint pour la connexion
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(fake_users_db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# Obtenir l'utilisateur actuel à partir du token JWT
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
        user = get_user(fake_users_db, username)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")

# Endpoint pour la page d'accueil
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, error: str = None):

@app.post("/predict", response_class=HTMLResponse)
async def predict_gravity(request: Request):
    df = pd.read_parquet('pd.read_parquet("data/dataset_Cramer.parquet")')
    prediction = model.predict(data)[0]
    return prediction 

# Endpoint pour télécharger un nouveau dataset et réentraîner le modèle
@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    print('Upload Data')

@app.post("/admin/retrain")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if file.filename.endswith('.parquet'):
        subprocess.run([
            "python", "accident_project/experiment.py",
            "--data", file.file,
        ])

       
     
# Lancer le serveur
# Pour lancer le serveur, exécutez dans votre terminal :
# uvicorn main:app --reload