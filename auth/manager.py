# -*- coding: utf-8 -*-
"""认证管理器：JSON 存储 + werkzeug pbkdf2:sha256 密码哈希 + 会话超时"""
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from werkzeug.security import generate_password_hash, check_password_hash

from auth.models import User
from core.exceptions import AuthError

DEFAULT_ADMIN = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


class AuthManager:
    def __init__(self, storage_dir: Path, session_timeout_minutes: int = 30):
        self.users_file = Path(storage_dir) / "users.json"
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_timeout_minutes = session_timeout_minutes
        self._users = {}
        self._current_user: Optional[User] = None
        self._login_time: Optional[float] = None
        self.load()

    # ---------- 持久化 ----------
    def load(self):
        if self.users_file.exists():
            try:
                raw = json.loads(self.users_file.read_text(encoding="utf-8"))
                self._users = {u: User.from_dict(u, d) for u, d in raw.items()}
            except (json.JSONDecodeError, OSError):
                self._users = {}

    def _save(self):
        self.users_file.write_text(
            json.dumps({u: user.to_dict() for u, user in self._users.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- 用户管理 ----------
    def ensure_default_admin(self) -> bool:
        """首次运行创建默认管理员 admin/admin123，返回是否新建"""
        if not self._users:
            self.add_user(DEFAULT_ADMIN, DEFAULT_ADMIN_PASSWORD, role="admin")
            return True
        return False

    def add_user(self, username: str, password: str, role: str = "user") -> Tuple[bool, str]:
        username = username.strip()
        if not username:
            return False, "用户名不能为空"
        if username in self._users:
            return False, f"用户 {username} 已存在"
        if len(password) < 6:
            return False, "密码长度至少 6 位"
        self._users[username] = User(
            username=username,
            password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
            role=role if role in ("admin", "user") else "user",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._save()
        return True, f"用户 {username} 创建成功"

    def delete_user(self, username: str) -> Tuple[bool, str]:
        if username not in self._users:
            return False, f"用户 {username} 不存在"
        if self._users[username].is_admin and sum(1 for u in self._users.values() if u.is_admin) <= 1:
            return False, "不能删除最后一个管理员"
        del self._users[username]
        self._save()
        return True, f"用户 {username} 已删除"

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        user = self._users.get(username)
        if user is None:
            return False, f"用户 {username} 不存在"
        if not check_password_hash(user.password_hash, old_password):
            return False, "原密码错误"
        if len(new_password) < 6:
            return False, "新密码长度至少 6 位"
        user.password_hash = generate_password_hash(new_password, method="pbkdf2:sha256")
        self._save()
        return True, "密码修改成功"

    def update_role(self, username: str, role: str) -> Tuple[bool, str]:
        user = self._users.get(username)
        if user is None:
            return False, f"用户 {username} 不存在"
        if role not in ("admin", "user"):
            return False, "非法角色"
        user.role = role
        self._save()
        return True, f"已更新 {username} 的角色为 {role}"

    def list_users(self) -> List[User]:
        return sorted(self._users.values(), key=lambda u: u.username)

    # ---------- 会话 ----------
    def login(self, username: str, password: str) -> Tuple[Optional[User], str]:
        user = self._users.get(username.strip())
        if user is None:
            return None, "用户名或密码错误"
        if not check_password_hash(user.password_hash, password):
            return None, "用户名或密码错误"
        self._current_user = user
        self._login_time = time.time()
        user.last_login = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return user, "登录成功"

    def login_as_guest(self) -> User:
        self._current_user = None
        self._login_time = time.time()
        return User(username="guest", password_hash="", role="guest")

    def logout(self):
        self._current_user = None
        self._login_time = None

    def is_authenticated(self) -> bool:
        if self._current_user is None or self._login_time is None:
            return False
        if self.session_timeout_minutes > 0 and (time.time() - self._login_time) > self.session_timeout_minutes * 60:
            self.logout()
            return False
        return True

    @property
    def current_user(self) -> Optional[User]:
        return self._current_user

    @property
    def current_username(self) -> str:
        return self._current_user.username if self._current_user else "guest"
