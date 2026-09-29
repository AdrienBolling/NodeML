"""Tests for ``nodeml.core.common.mixins.mixin``."""

from __future__ import annotations

from nodeml.core.common.mixins.mixin import MixinSettings


class TestMixinSettings:
    def test_each_instance_has_its_own_id(self) -> None:
        assert MixinSettings()._id != MixinSettings()._id
