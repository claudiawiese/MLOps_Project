# main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from endpoints.auth import oauth2_scheme, get_user, create_access_token, get_current_user
from endpoints.train import router as train_router
from endpoints.predict import router as predict_router
from endpoints.auth import router as auth_router

# Create FastAPI app
app = FastAPI()

# Include routers
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(auth_router)

# Home endpoint
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, error: str = None):
    return "Hello Home"


#Code De Clement pour la BDD_users
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")

# Crée l'engine asynchrone
engine = create_async_engine(DATABASE_URL, echo=True)

# Crée une session asynchrone
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dépendance de session pour l'injection dans les routes
async def get_db():
    async with async_session() as session:
        yield session
