import time
from datetime import timedelta


def timing(log_file: str = None):
    """
    Measures the execution time of a function and appends it to a log file.

    :param str log_file: The name of the log file to append execution times (optional).
    """

    def decorator(f):
        """
        A decorator for measuring the execution time of a function.

        :param callable f: The function to be measured.

        :return: The wrapped function.
        """

        # @wraps(f)
        def wrap(*args, **kwargs):
            """
            Wrapper function that measures execution time and logs it.

            :param args: Positional arguments for the function.
            :param kwargs: Keyword arguments for the function.

            :return: The result of the wrapped function.
            """
            tracker = args[0]

            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed_time = (time.time() - start_time)

            tracker.logger.info(msg=f'{f.__name__} function took {str(timedelta(seconds=elapsed_time))}')

            if log_file is not None:
                with open(log_file, 'a') as file:
                    file.write(f'{f.__module__},{f.__name__},{elapsed_time}\n')

            return result

        return wrap

    return decorator


def suppress_tracking(f):
    """
    Disables the tracker tracking before executing the function.

    :param callable f: The function to be executed with tracking disabled.

    :return: The wrapped function.
    """

    def wrap(*args, **kwargs):
        """
        Wrapper function that temporarily disables tracking and then reverts it.

        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.

        :return: The result of the wrapped function.
        """
        tracker = args[0]
        temp1, temp2 = tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking
        tracker.enable_dataframe_warning_msg, tracker.dataframe_tracking = False, False
        result = f(*args, **kwargs)
        tracker.dataframe_tracking, tracker.enable_dataframe_warning_msg = temp1, temp2

        return result

    return wrap


class Singleton:
    """
    A decorator for implementing the Singleton design pattern.

    :param callable cls: The class to be turned into a Singleton.
    """

    def __init__(self, cls) -> None:
        self.cls = cls
        self.instance = None

    def __call__(self, *args, **kwargs):
        """
        Overrides the call method to create a Singleton instance of the class.

        :param args: Positional arguments for the class constructor.
        :param kwargs: Keyword arguments for the class constructor.

        :return: The Singleton instance of the class.
        """
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance
