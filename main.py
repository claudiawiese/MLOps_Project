# main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from endpoints.auth import oauth2_scheme, get_user, create_access_token, get_current_user
from endpoints.train import router as train_router
from endpoints.predict import router as predict_router

# Create FastAPI app
app = FastAPI()

# Include routers
app.include_router(train_router)
app.include_router(predict_router)

# Home endpoint
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request, error: str = None):
    return "Hello Home"

# Login endpoint
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