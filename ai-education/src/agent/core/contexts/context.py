from contextvars import ContextVar
from typing import Optional

from agent.core.entities.user_models import User


class UserContext:
    current_user: ContextVar[Optional[User]] = ContextVar('current_user', default=None)

    @classmethod
    def get_current_user(cls) -> Optional[User]:
        return cls.current_user.get()

    @classmethod
    def set_current_user(cls, user: Optional[User]) -> None:
        cls.current_user.set(user)

    @classmethod
    def clear_current_user(cls) -> None:
        cls.current_user.set(None)