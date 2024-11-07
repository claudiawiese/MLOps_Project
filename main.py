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