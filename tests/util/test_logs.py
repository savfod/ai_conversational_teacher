import time

from conversa.util.logs import get_logger, log_function_duration


def test_log_function_duration(tmp_path):
    log_path = tmp_path / "test_duration.log"
    get_logger("conversa.timed_func").add(log_path)

    # @log_function_duration(name="conversa.timed_func")
    # __module__ should be altered before running the decorator
    # so it has to be applied after the __module__ change
    def timed_func() -> int:
        time.sleep(0.1)
        return 42

    timed_func.__module__ = "conversa.timed_func"
    timed_func = log_function_duration(name="conversa.timed_func")(timed_func)
    result = timed_func()

    assert result == 42
    assert "timed_func completed in:" in open(log_path).read()
