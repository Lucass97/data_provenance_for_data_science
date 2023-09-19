import time
from datetime import timedelta


"""
def timing(f, log_file):


    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = (time.time() - start_time)
        tracker = args[0]
        
        msg=f'{f.__name__} function took {str(timedelta(seconds=elapsed_time))}'

        tracker.logger.info(msg=msg)
        with open(log_file, 'a') as file:
            file.write(f'{msg}')

        return result

    return wrap
"""

def timing(log_file: str = None):
    """
    Measures the execution time of a function and appends it to a log file.
    """

    def decorator(f):
        
        def wrap(*args, **kwargs):

            tracker = args[0]

            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed_time = (time.time() - start_time)

            tracker.logger.info(msg=f'{f.__name__} function took {str(timedelta(seconds=elapsed_time))}')

            if log_file is not None:
                with open(log_file, 'a') as file:
                    file.write(f'{f.__name__},{elapsed_time}\n')

            return result

        return wrap

    return decorator


def suppress_tracking(f):
    """
    Disables the tracker tracking before executing the function.
    """

    def wrap(*args, **kwargs):
        tracker = args[0]
        temp1, temp2 = tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking
        tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking = False, False
        result = f(*args, **kwargs)
        tracker.dataframe_tracking, tracker.enable_dataframe_warning_msg = temp1, temp2

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
