# -*- coding: utf-8 -*-
"""用户数据模型"""
from dataclasses import dataclass, field


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "user"          # admin | user
    created_at: str = ""
    last_login: str = ""

    def to_dict(self) -> dict:
        return {
            "password_hash": self.password_hash,
            "role": self.role,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }

    @classmethod
    def from_dict(cls, username: str, data: dict) -> "User":
        return cls(
            username=username,
            password_hash=data.get("password_hash", ""),
            role=data.get("role", "user"),
            created_at=data.get("created_at", ""),
            last_login=data.get("last_login", ""),
        )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"
