from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.models import ROLE_ADMIN, User
from app.db.session import get_db

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status.HTTP_401_UNAUTHORIZED,
    "Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise _UNAUTHORIZED
    try:
        payload = decode_access_token(creds.credentials)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError):
        raise _UNAUTHORIZED
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user  # role is read from the DB, never trusted from the token


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != ROLE_ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator access required")
    return user
