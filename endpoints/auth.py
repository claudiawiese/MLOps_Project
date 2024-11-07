# auth.py
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict

# Security configurations
SECRET_KEY = "votre_clé_secrète"  # Replace with your secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fake users database
fake_users_db = {
    "admin": {"username": "admin", "role": "admin", "hashed_password": pwd_context.hash("admin123")},
    "manager": {"username": "manager", "role": "manager", "hashed_password": pwd_context.hash("manager123")},
    "user": {"username": "user", "role": "user", "hashed_password": pwd_context.hash("user123")}
}

class User(BaseModel):
    username: str
    role: str

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db: Dict[str, Dict], username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = get_user(fake_users_db, username)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

async def get_admin_user(current_user: User = Security(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user
