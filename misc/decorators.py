import time
from datetime import timedelta


def timing(f):
    """
    Measures the execution time of a function.
    """

    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = (time.time() - start_time)
        tracker = args[0]
        tracker.logger.info(msg=f' {f.__name__} function took {str(timedelta(seconds=elapsed_time))}')

        return result

    return wrap


def suppress_tracking(f):
    """
    Disables the tracker tracking before executing the function.
    """

    def wrap(*args, **kwargs):
        tracker = args[0]
        tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking, = False, False
        result = f(*args, **kwargs)
        tracker.dataframe_tracking, tracker.enable_dataframe_warning_msg = True, True

        return result

    return wrap


class Singleton:
    def __init__(self, cls) -> None:
        self.cls = cls
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance
