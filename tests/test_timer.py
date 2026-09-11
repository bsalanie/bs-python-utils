import time

import pytest

from bs_python_utils.core.Timer import Timer, TimerError, timeit


class TestTimer:
    """Tests for the Timer context manager and manual timing."""

    def test_context_manager_basic(self, capsys):
        """Timer works as a context manager and prints default message."""
        with Timer():
            time.sleep(0.01)

        captured = capsys.readouterr()
        assert "Elapsed time:" in captured.out
        assert "seconds" in captured.out

    def test_context_manager_captures_elapsed_time(self):
        """Timer context manager automatically stops when exiting."""
        with Timer(logger=None) as timer:
            time.sleep(0.01)

        # Context manager exit should have stopped the timer
        assert timer._start_time is None  # Timer should be stopped

    def test_manual_start_stop(self, capsys):
        """Manual start() and stop() calls work correctly."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()

        assert elapsed > 0.01
        captured = capsys.readouterr()
        assert "Elapsed time:" in captured.out

    def test_custom_text_format(self, capsys):
        """Custom text format is used for output."""
        custom_text = "Took {:0.2f}s to execute"
        with Timer(text=custom_text):
            time.sleep(0.01)

        captured = capsys.readouterr()
        assert "Took" in captured.out
        assert "s to execute" in captured.out

    def test_logger_none_suppresses_output(self, capsys):
        """Setting logger=None suppresses output."""
        with Timer(logger=None):
            time.sleep(0.01)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_custom_logger(self):
        """Custom logger function receives formatted output."""
        output_list = []
        with Timer(logger=output_list.append):
            time.sleep(0.01)

        assert len(output_list) == 1
        assert "Elapsed time:" in output_list[0]

    def test_stop_returns_elapsed_time(self):
        """stop() returns the elapsed time as a float."""
        timer = Timer(logger=None)
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()

        assert isinstance(elapsed, float)
        assert elapsed > 0.01

    def test_named_timer_accumulates(self, capsys):
        """Named timers accumulate elapsed time across multiple calls."""
        Timer.timers.clear()

        timer1 = Timer(name="test_op", logger=None)
        timer1.start()
        time.sleep(0.01)
        timer1.stop()

        timer2 = Timer(name="test_op", logger=None)
        timer2.start()
        time.sleep(0.01)
        timer2.stop()

        assert "test_op" in Timer.timers
        assert Timer.timers["test_op"] > 0.02

    def test_named_timer_initialized_in_dict(self):
        """Named timer is initialized to 0 in timers dict on creation."""
        Timer.timers.clear()
        Timer(name="new_timer")

        assert "new_timer" in Timer.timers
        assert Timer.timers["new_timer"] == 0

    def test_unnamed_timer_not_in_dict(self):
        """Unnamed timers are not added to the timers dict."""
        Timer.timers.clear()
        Timer(logger=None)

        assert len(Timer.timers) == 0

    def test_start_when_already_running_raises_error(self):
        """Calling start() when timer is running raises TimerError."""
        timer = Timer(logger=None)
        timer.start()

        with pytest.raises(TimerError, match="Timer is running"):
            timer.start()

        timer.stop()

    def test_stop_when_not_running_raises_error(self):
        """Calling stop() when timer is not running raises TimerError."""
        timer = Timer()

        with pytest.raises(TimerError, match="Timer is not running"):
            timer.stop()

    def test_context_manager_with_exception_still_stops(self):
        """Timer still stops even if exception occurs in with block."""
        timer = None
        try:
            with Timer(logger=None) as t:
                timer = t
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert timer._start_time is None

    def test_elapsed_time_is_reasonable(self):
        """Elapsed time measurement is reasonably accurate."""
        timer = Timer(logger=None)
        timer.start()
        sleep_duration = 0.02
        time.sleep(sleep_duration)
        elapsed = timer.stop()

        # Allow some tolerance for system variations
        assert elapsed > sleep_duration * 0.9
        assert elapsed < sleep_duration * 2.0

    def test_enter_returns_self(self):
        """__enter__ returns the Timer instance itself."""
        timer = Timer(logger=None)
        returned = timer.__enter__()

        assert returned is timer
        timer.stop()

    def test_multiple_named_timers_independent(self, capsys):
        """Multiple named timers maintain independent accumulations."""
        Timer.timers.clear()

        timer_a = Timer(name="timer_a", logger=None)
        timer_a.start()
        time.sleep(0.01)
        timer_a.stop()

        timer_b = Timer(name="timer_b", logger=None)
        timer_b.start()
        time.sleep(0.01)
        timer_b.stop()

        assert Timer.timers["timer_a"] > 0.01
        assert Timer.timers["timer_b"] > 0.01
        # They should be approximately equal but may have small differences
        assert abs(Timer.timers["timer_a"] - Timer.timers["timer_b"]) < 0.01


class TestTimeitDecorator:
    """Tests for the timeit function decorator."""

    def test_timeit_prints_execution_time(self, capsys):
        """timeit decorator prints function execution time."""

        @timeit
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()

        assert result == 42
        captured = capsys.readouterr()
        assert "slow_function executed in" in captured.out
        assert "seconds" in captured.out

    def test_timeit_preserves_function_name(self):
        """timeit decorator preserves the original function name."""

        @timeit
        def my_function():
            return "result"

        assert my_function.__name__ == "my_function"

    def test_timeit_returns_correct_result(self):
        """timeit decorator returns the function's return value unchanged."""

        @timeit
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_timeit_with_args_and_kwargs(self, capsys):
        """timeit works with functions that have arguments and keyword arguments."""

        @timeit
        def process(a, b, multiplier=1):
            return (a + b) * multiplier

        result = process(2, 3, multiplier=2)

        assert result == 10
        captured = capsys.readouterr()
        assert "process executed in" in captured.out
