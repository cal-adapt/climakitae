"""
Notes about this logger (Updated 5/1/24):

This logger is to be used for developers who want to see a comprehensive view of the function call stacks of any functions
within `climakitae`. In order to use this logger, you must do the following:

    from climakitae.util.dev_logging import _enable_lib_logging, _disable_lib_logging
    from climakitae import YOUR_MODULE_HERE

    _enable_lib_logging(YOUR_MODULE_HERE)
    # To disable it after you're done viewing the logger: _disable_lib_logging(YOUR_MODULE_HERE)

Once you've enabled the logger, you should be able to view comprehensive call-stacks and runtimes for functions within that module,
and other functions within submodules or subclasses within that module. The logger does its best to try and find all `climakitae` functions
that are within this module or any sub-modules (while avoiding putting repeated wrappers), but it is best practice to wrap each module that
you are interested in logging separately. I.e.:

    _enable_lib_logging(YOUR_MODULE_HERE_1)
    _enable_lib_logging(YOUR_MODULE_HERE_2)


Some limitations of this logger include:
    1. If you interrupt function calls that are wrapped with this logger, it will not reset the `indentation_level` of printed calls.
    What would need to happen in order to resolve this is for you to disable and re-enable the library logger.

"""

import time
import types
import importlib

# Controls the amount of indentation for library logging
indentation_level = 0


def _log(func):
    """
    Wraps around existing functions, adding print statements upon execution and the amount of time it takes for execution. Allows function's call-stack to be viewed for all sub-functions that also have this wrapper.
    """

    def _wrapper(*args, **kwargs):
        """Wraps timer and print statement around functions if `lib_log_enabled` is True, otherwise
        just return the function's result."""
        global indentation_level
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

    return _wrapper


def _enable_lib_logging(obj):
    """Adds the `log` wrapper to all functions and sub-classes within the given module or class.

    Parameters
    ----------
    obj: types.ModuleType or type (module or class)
    """
    # Check if the passed in object is a module or a class
    if isinstance(obj, types.ModuleType) or isinstance(obj, type):
        for name in dir(obj):
            res = getattr(obj, name)

            # Initial logic to prevent loggers from double wrapping functions
            if "__name__" in dir(res):
                if res.__name__ == "wrapper":
                    continue

            # Do not add loggers to any built-in functions
            if not name.startswith("__") and not name.endswith("__"):

                # Only add loggers to objects that are function types
                if isinstance(res, types.FunctionType):

                    # Only add loggers to functions within AE
                    if "climakitae" in res.__module__:
                        setattr(obj, name, _log(res))

                # Check if the object is a class type, is not the literal string '__class__', and is created within climakitae.
                elif (
                    isinstance(res, type)
                    and name != "__class__"
                    and res.__module__[:10] == "climakitae"
                ):
                    # Recursively add logging wrapper to any classes within the passed in module/class.
                    _enable_lib_logging(res)
    else:
        print("Error: Current object is not a module object.")


def _disable_lib_logging(module):
    """
    Removes the `log` wrapper to all functions within the given module.
    """
    # Currently, in order to remove the logger from all decorated functions, you will need to reload the module.
    # I have tried to reference a `__wrapped__` attribute on wrapped functions, but they don't seem to exist.
    importlib.reload(module)
    return
