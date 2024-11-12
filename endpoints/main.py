from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from models import User
from schemas import UserCreate
from database import get_db
from utils import hash_password

app = FastAPI()

@app.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Vérifie si l'utilisateur existe déjà
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Crée un nouvel utilisateur avec le mot de passe haché
    hashed_password = hash_password(user.password)
    new_user = User(username=user.username, role=user.role, password=hashed_password)

    # Ajoute l'utilisateur à la base de données
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return user
