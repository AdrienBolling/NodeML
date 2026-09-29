"""Base settings and utilities for framework mixins."""

import uuid

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


class MixinSettings(BaseSettings):
    """Base configuration for Mixins."""

    # default_factory gives each instance its own UUID.
    _id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)
