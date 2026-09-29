"""Tests for :class:`nodeml.core.pipeline.runners.wrappers.wrapper.PipelineRunnerWrapper`.

The wrapper is an abstract base class, so we drive it through a trivial
identity subclass whose only job is to forward every call to the
underlying :class:`SmartRunner`.  That is enough to cover the property
delegation + method forwarding surface.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.pipeline.runners.pipeline_runner import PipelineRunner
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from nodeml.core.pipeline.runners.wrappers.wrapper import PipelineRunnerWrapper
from tests.shims.pipelines import build_source_model_sink_pipeline


class _WrapperConfig(BaseModel):
    pass


class _IdentityWrapper(PipelineRunnerWrapper[object, object, _WrapperConfig]):
    """A concrete wrapper that merely forwards every call."""

    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        *,
        config: _WrapperConfig | None = None,
    ) -> None:
        super().__init__(pipeline_runner, config=config or _WrapperConfig())


class TestWrapperDelegation:
    def test_wrapper_exposes_underlying_pipeline(self, regression_dataset) -> None:
        pipeline = build_source_model_sink_pipeline()
        runner = SmartRunner(pipeline)
        wrapped = _IdentityWrapper(runner)

        assert wrapped.pipeline is runner.pipeline
        assert wrapped.pipeline_runner is runner
        assert wrapped.unwrapped is runner

    def test_wrapper_forwards_train_and_infer(self, regression_dataset) -> None:
        (X_pair, y_pair, _) = regression_dataset
        pipeline = build_source_model_sink_pipeline()
        runner = SmartRunner(pipeline)
        wrapped = _IdentityWrapper(runner)

        wrapped.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        preds: Mapping[str, tuple[ArrayLike, DataContext]] = wrapped.infer(
            input_data={"source": {"X": X_pair, "y": y_pair}}
        )
        assert "pred" in preds

    def test_wrapper_nested_unwraps_to_root(self, regression_dataset) -> None:
        pipeline = build_source_model_sink_pipeline()
        runner = SmartRunner(pipeline)
        inner = _IdentityWrapper(runner)
        outer = _IdentityWrapper(inner)

        assert outer.unwrapped is runner
