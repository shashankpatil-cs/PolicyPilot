from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.ratelimit import check_login_allowed, clear_login_failures, record_login_failure
from app.core.security import (
    BCRYPT_MAX_BYTES,
    burn_password_check,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models import ROLE_ADMIN, ROLE_USER, User
from app.db.session import get_db
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

_INVALID_LOGIN = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


def _login(request: Request, body: LoginRequest, db: Session, *, admin_only: bool) -> TokenResponse:
    username = body.username.strip().lower()
    client = request.client.host if request.client else "unknown"
    key = f"{'admin' if admin_only else 'user'}:{client}:{username}"
    check_login_allowed(key)

    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        burn_password_check(body.password)
        record_login_failure(key)
        raise _INVALID_LOGIN
    if not user.is_active or not verify_password(body.password, user.password_hash):
        record_login_failure(key)
        raise _INVALID_LOGIN
    if admin_only and user.role != ROLE_ADMIN:
        record_login_failure(key)
        raise _INVALID_LOGIN

    clear_login_failures(key)
    return _token_response(user)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    username = body.username.strip().lower()
    if not re.fullmatch(settings.USERNAME_REGEX, body.username.strip()):
        raise HTTPException(422, "Username format is not allowed")
    if len(body.password) < settings.PASSWORD_MIN_LENGTH:
        raise HTTPException(
            422,
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
        )
    if len(body.password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise HTTPException(
            422, f"Password must be at most {BCRYPT_MAX_BYTES} bytes"
        )
    if db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Username is not available")

    user = User(username=username, password_hash=hash_password(body.password), role=ROLE_USER, is_active=True)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Username is not available")
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return _login(request, body, db, admin_only=False)


@router.post("/admin/login", response_model=TokenResponse)
def admin_login(request: Request, body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return _login(request, body, db, admin_only=True)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
