import time
import types
from functools import wraps
import importlib

# Define global variables to control logging
lib_log_enabled = False  # For developers

# Controls the amount of indentation for library logging
indentation_level = 0


def enable_lib_logging():
    """
    Enables library logging.
    """
    global lib_log_enabled
    if not lib_log_enabled:
        lib_log_enabled = True


def disable_lib_logging():
    """
    Disables library logging.
    """
    global lib_log_enabled
    if lib_log_enabled:
        lib_log_enabled = False


def log(func):
    """
    Wraps around existing functions, adding print statements upon execution and the amount of time it takes for execution. Allows function's call-stack to be viewed for all sub-functions that also have this wrapper.
    """
    global lib_log_enabled

    def wrapper(*args, **kwargs):
        """Wraps timer and print statement around functions if `lib_log_enabled` is True, otherwise
        just return the function's result."""
        global indentation_level
        if lib_log_enabled:
            start_time = time.time()
            print("    " * indentation_level + f"Executing function: {func.__name__}")
            indentation_level += 1
            results = func(*args, **kwargs)
            indentation_level -= 1
            end_time = time.time()
            print(
                "    " * indentation_level
                + f"Execution time for {func.__name__}: {end_time - start_time:.4g}"
            )
            return results
        return func(*args, **kwargs)

    return wrapper


def add_log_wrapper(obj):
    """
    Adds the `log` wrapper to all functions within the given module.
    """
    import pdb; pdb.set_trace()
    # Check if the module 
    if isinstance(obj, types.ModuleType) or isinstance(obj, type):
        for name in dir(obj):
            res = getattr(obj, name)
            if isinstance(res, type):
                add_log_wrapper(res)
            elif isinstance(res, types.FunctionType):
                setattr(obj, name, log(res))
    else:
        print("Error: Current object is not a module object.")


def remove_log_wrapper(module):
    """
    Removes the `log` wrapper to all functions within the given module.
    """
    # Currently, in order to remove the logger from all decorated functions, you will need to reload the module.
    # I have tried to reference a `__wrapped__` attribute on wrapped functions, but they don't seem to exist.
    importlib.reload(module)
    return
