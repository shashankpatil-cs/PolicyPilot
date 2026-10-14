"""Print a bcrypt hash for ADMIN_PASSWORD_HASH.   python scripts/hash_password.py"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.security import hash_password  # noqa: E402

pw = getpass.getpass("New admin password: ")
if pw != getpass.getpass("Repeat: "):
    sys.exit("Passwords do not match")
print(hash_password(pw))
