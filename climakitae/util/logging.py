import logging
import sys
import time
import types
from functools import wraps
import importlib

# Define global variables to control logging
app_log_enabled = False  # For users
lib_log_enabled = False  # For developers

# Controls the amount of indentation for library logging
indentation_level = 0

# Instantiating loggers
logger = logging.getLogger("Climakitae Back-end Debugger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))


# Creating an additional logger for adding new lines around logger when enabling full library logger
# (i.e. view logger lines as well as call stack and runtimes
class NewlineHandler(logging.Handler):
    def emit(self, record):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        msg = self.format(record)
        msg = f"\n{timestamp} - {record.name} - {msg}\n"
        print(msg)


# Create the extra handlers for new lines
newline_handler = NewlineHandler()
newline_handler.setLevel(logging.DEBUG)


### Functions to enable or disable logging in notebooks


def enable_lib_logging():
    """
    Enables library logging.
    """
    global lib_log_enabled
    if not lib_log_enabled:
        lib_log_enabled = True
        # app_log_enabled = False
        # logger.addHandler(newline_handler)
        # logger.removeHandler(handler)


def disable_lib_logging():
    """
    Disables library logging.
    """
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
            print("  " * indentation_level + f"Executing function: {func.__name__}")
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


def add_log_wrapper(module):
    """
    Adds the `log` wrapper to all functions within the given module.
    """
    # TODO: Calvin- This function only works when it exists in a notebook, not here.
    # I believe this has something to do with how modules are referenced in a notebook vs in the back-end.
    if isinstance(module, types.ModuleType):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, types.FunctionType):
                setattr(module, name, log(obj))
    else:
        print("Error: Current object is not a module object.")


def remove_log_wrapper(module):
    """
    Removes the `log` wrapper to all functions within the given module.
    """
    # TODO: Calvin - I can't seem to remove the decorator on the subfunctions.
    # I am told to get each function's __wrapped__ attribute, but they don't exist on the objects.
    # Currently, the only way to remove all log wrappers to a module is to 1. reload the module 2. restart the kernel.

    # if isinstance(agnostic, types.ModuleType):
    #     for name in dir(agnostic):
    #         obj = getattr(agnostic, name)
    #         if isinstance(obj, types.FunctionType):
    #             # Check if the function is wrapped with 'log'
    #             if hasattr(obj, '__wrapped__'):
    #                 setattr(agnostic, name, obj.__wrapped__)

    importlib.reload(module)
    return


def kill_loggers():
    """
    Kills all active handlers for current logger.
    """
    global logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
