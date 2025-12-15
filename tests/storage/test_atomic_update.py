"""

ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import pytest

from src.storage.atomic_update import AtomicUpdateError, AtomicUpdateGroup


def test_atomic_group_executes_steps_in_order() -> None:
    """
    test_atomic_group_executes_steps_in_order: Function description.
    :param:
    :returns:
    """

    group = AtomicUpdateGroup.begin("key")
    group.add_step("one", lambda ctx: ctx.setdefault("log", []).append("one"))
    group.add_step("two", lambda ctx: ctx.setdefault("log", []).append("two"))

    ctx = group.execute()
    assert ctx["log"] == ["one", "two"]


def test_atomic_group_rolls_back_completed_steps() -> None:
    """
    test_atomic_group_rolls_back_completed_steps: Function description.
    :param:
    :returns:
    """

    group = AtomicUpdateGroup.begin("key")
    group.add_step("set", lambda ctx: ctx.__setitem__("value", 1), undo=lambda ctx: ctx.pop("value", None))
    group.add_step("boom", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(AtomicUpdateError, match="Atomic update group failed"):
        group.execute()

    assert "value" not in group.context


def test_atomic_group_prevents_appending_after_execute() -> None:
    """
    test_atomic_group_prevents_appending_after_execute: Function description.
    :param:
    :returns:
    """

    group = AtomicUpdateGroup.begin("key")
    group.execute()
    with pytest.raises(RuntimeError, match="append steps after execute"):
        group.add_step("x", lambda ctx: None)


def test_atomic_group_serializes_concurrent_executes() -> None:
    """
    test_atomic_group_serializes_concurrent_executes: Function description.
    :param:
    :returns:
    """

    group1 = AtomicUpdateGroup.begin("same-key")
    group2 = AtomicUpdateGroup.begin("same-key")
    timeline: List[str] = []
    gate = threading.Event()

    def step1(ctx: Dict[str, Any]) -> None:
        """
        step1: Function description.
        :param ctx:
        :returns:
        """

        timeline.append("t1-enter")
        gate.set()
        time.sleep(0.05)
        timeline.append("t1-exit")

    def step2(ctx: Dict[str, Any]) -> None:
        """
        step2: Function description.
        :param ctx:
        :returns:
        """

        gate.wait(timeout=1.0)
        timeline.append("t2-enter")
        timeline.append("t2-exit")

    group1.add_step("slow", step1)
    group2.add_step("fast", step2)

    t1 = threading.Thread(target=group1.execute)
    t2 = threading.Thread(target=group2.execute)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert "t1-enter" in timeline and "t2-enter" in timeline
    assert timeline.index("t1-exit") < timeline.index("t2-enter")
