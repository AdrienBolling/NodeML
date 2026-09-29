"""Semantic version model."""

from pydantic import BaseModel


class Version(BaseModel):
    """Semantic version, for example the version of a pipeline."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Return the version as a string in the format 'major.minor.patch'."""
        return f"{self.major}.{self.minor}.{self.patch}"
