import logging
import functools

__version__ = "0.0.7"


def trycatch(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception('Exception occurred: [{}]'.format(e))
    return wrapper


def benchmark(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from time import time
        start_time = time()
        func(*args, **kwargs)
        end_time = time()
        logging.info('Time taken by the function is [{time}] sec'.format(func=func, time=end_time-start_time))
    return wrapper

def run_in_thread(func):
    """
    Handy decorator for running a function in thread.
    Description:
        - Using standard threading.Thread for creating thread
        - Can pass args and kwargs to the function
        - Will start a thread but will give no control over it
    Use:
        Printing ('Siddhesh',) from thread
        Thread started for function <function display at 0x7f1d60f7cb90>
        Printing ('Siddhesh',) from thread
        Printing ('Siddhesh',) from thread
        Printing ('Siddhesh',) from thread
        Printing ('Siddhesh',) from thread

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import threading
        threading.Thread(target=func, args=(args, kwargs)).start()
        logging.info('Thread started for function {}'.format(func))
    return wrapper


def create_n_threads(thread_count=1):
    """
    Handy decorator for creating multiple threads of a single function
    Description:
        - Using standard threading.Thread for thread creation
        - Can pass args and kwargs to the function
        - Will start number of threads based on the count specified while decorating
    Use:
    """
    def wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            for i in range(thread_count):
                threading.Thread(target=func, args=(args, kwargs)).start()
                logging.info('Thread started for function {}'.format(func))
        return wrapper
    return wrapper
