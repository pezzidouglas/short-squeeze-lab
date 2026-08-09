"""
Compatibility shim for the 1.x module layout.

The scoring engine now lives in `squeezelab.engine`. It gained an off-exchange
flow component, weight presets and per-component contributions; the original
40/25/20/15 model is preserved as the `classic` preset, and the legacy keyword
arguments below select it automatically.
"""

from squeezelab.engine import COMPONENTS, ComponentResult, SqueezeEngine

__all__ = ["COMPONENTS", "ComponentResult", "SqueezeEngine"]
