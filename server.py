import os
import traceback
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from starlette.requests import Request

from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError

from passlib.context import CryptContext
from jose import jwt

# ========= Config =========
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "quantumquant")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_TO_LONG_RANDOM_SECRET")
JWT_ALG = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# Keep bcrypt for now since your project already used it
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ========= App =========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your project folder at /static (HTML/CSS/JS/images)
# Example: /static/signup.html
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

client = AsyncIOMotorClient(MONGO_URL)
db = client[MONGO_DB]

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# ========= HTML routes =========
@app.get("/")
async def home():
    # choose your landing page
    return FileResponse("index.html")

@app.get("/login")
async def login_page():
    return FileResponse("login.html")

@app.get("/signup")
async def signup_page():
    return FileResponse("signup.html")


# ========= Helpers =========
def hash_password(password: str) -> str:
    return pwd.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd.verify(password, password_hash)

def make_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=JWT_EXPIRE_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


# ========= Models =========
class SignupIn(BaseModel):
    firstName: str = Field(min_length=1, max_length=80)
    lastName: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    plan: str = "free"

class LoginIn(BaseModel):
    email: EmailStr
    password: str


# ========= Startup =========
@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    print("[mongo] ensured users.email unique index")


# ========= API =========
@app.get("/health")
async def health():
    return {"ok": True, "db": MONGO_DB, "mongo_url": MONGO_URL}

@app.post("/api/signup")
async def signup(body: SignupIn):
    email = body.email.lower().strip()

    doc = {
        "email": email,
        "first_name": body.firstName.strip(),
        "last_name": body.lastName.strip(),
        "password_hash": hash_password(body.password),
        "plan": body.plan,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }

    try:
        res = await db.users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Email already exists.")

    user_id = str(res.inserted_id)
    initials = (doc["first_name"][:1] + doc["last_name"][:1]).upper()
    token = make_token(user_id, email)
    return {"token": token, "user": {"id": user_id, "email": email, "initials": initials}}

@app.post("/api/login")
async def login(body: LoginIn):
    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email})

    if not user or not user.get("is_active", False):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    user_id = str(user["_id"])
    initials = (user["first_name"][:1] + user["last_name"][:1]).upper()
    token = make_token(user_id, email)

    return {"token": token, "user": {"id": user_id, "email": email, "initials": initials}}

app.mount("/", StaticFiles(directory=".", html=True), name="site")
