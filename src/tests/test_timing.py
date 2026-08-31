"""
Unit tests for timing.py
"""

import time

import pytest

import timing


@pytest.fixture(autouse=True)
def clean_registry():
    timing.reset()
    yield
    timing.reset()


def test_record_accumulates_counts_and_durations():
    for _ in range(3):
        with timing.record("stage-a"):
            time.sleep(0.005)

    total, count = timing._registry["stage-a"]
    assert count == 3
    assert total >= 0.015


def test_record_accumulates_even_when_body_raises():
    with pytest.raises(RuntimeError):
        with timing.record("stage-raise"):
            raise RuntimeError("boom")

    _, count = timing._registry["stage-raise"]
    assert count == 1


def test_report_contains_stages_sorted_by_total_descending():
    with timing.record("fast"):
        pass
    with timing.record("slow"):
        time.sleep(0.01)

    report = timing.report()
    assert "fast" in report
    assert "slow" in report
    assert report.index("slow") < report.index("fast")


def test_reset_clears_registry():
    with timing.record("stage-b"):
        pass
    timing.reset()

    assert "stage-b" not in timing.report()
    assert timing.report() == "no timings recorded"
