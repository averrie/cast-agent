"""Agent loops - the three CAST evaluation conditions."""

from cast_agent.loops.stuffed import run_stuffed
from cast_agent.loops.react import run_react
from cast_agent.loops.cast import run_cast

__all__ = ["run_stuffed", "run_react", "run_cast"]
