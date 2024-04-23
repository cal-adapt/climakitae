import logging
import inspect
import functools
import sys
import time

# Define global variables to control logging
app_log_enabled = False # For users
lib_log_enabled = False # For developers

# Controls the amount of indentation for library logging
indentation_level = 0

# Let user see what the current logging status is
current_logging_status = lambda: logging_enabled

# Instantiating loggers
logger = logging.getLogger("Climakitae Back-end Debugger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

# Creating an additional logger for adding new lines around logger when enabling full library logger
# (i.e. view logger lines as well as call stack and runtimes
class NewlineHandler(logging.Handler):
    def emit(self, record):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
        msg = self.format(record)
        msg = f"\n{timestamp} - {record.name} - {msg}\n"
        print(msg)

# Create the extra handlers for new lines
newline_handler = NewlineHandler()
newline_handler.setLevel(logging.DEBUG)
        

### Functions to enable or disable logging in notebooks
    
def enable_app_logging():
    """
    Enables application logging by adding a handler.
    """"
    global app_log_enabled
    if not app_log_enabled:
        app_log_enabled = True
        logger.addHandler(handler)


def disable_logging():
    """
    Disables logging by removing application logger handler.
    """"
    global app_log_enabled
    if app_log_enabled:
        app_log_enabled = False
        logger.removeHandler(handler)
        

def enable_lib_logging():
    """
    Enables library logging.
    """"
    global lib_log_enabled
    if not lib_log_enabled:
        lib_log_enabled = True
        # app_log_enabled = False
        # logger.addHandler(newline_handler)
        # logger.removeHandler(handler)
        

def disable_lib_logging():
    """
    Disables library logging.
    """"
    global lib_log_enabled
    if lib_log_enabled:
        lib_log_enabled = False
        # logger.removeHandler(newline_handler)
        
        
def log(func):
    """
    Wraps around existing functions, adding print statements upon execution and the amount of time it takes for execution. Allows function's call-stack to be viewed for all sub-functions that also have this wrapper.
    """
    global lib_log_enabled, logger

    def wrapper(*args, **kwargs):
        """Wraps timer and print statement around functions if `lib_log_enabled` is True, otherwise
        just return the function's result."""
        global indentation_level
        if lib_log_enabled:
            start_time = time.time()
            print(
                "  " * indentation_level + f"Executing function: {func.__name__}"
            )
            indentation_level += 1
            results = func(*args, **kwargs)
            indentation_level -= 1
            end_time = time.time()
            print(
                "  " * indentation_level
                + f"Execution time for {func.__name__}: {end_time - start_time:.4g}"
            )
            return results
        return func(*args, **kwargs)

    return wrapper


def kill_loggers():
    """
    Kills all active handlers for current logger.
    """
    global logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)