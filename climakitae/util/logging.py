import logging
import inspect
import functools
import sys
import time


# Define a global flag to control logging
logging_enabled = False

# Define a list of allowed module names
allowed_modules = ["climakitae"]

current_logging_status = lambda: logging_enabled

# Trying to use logger
logger = logging.getLogger("Climakitae Back-end Debugger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(message)s")
)
logger.addHandler(handler)

indentation_level = 0


def enable_logging():
    """
    Enable logging for library functions, methods, and class instantiations.
    """
    global logging_enabled
    logging_enabled = True


def disable_logging():
    """
    Disable logging for library functions, methods, and class instantiations.
    """
    global logging_enabled
    logging_enabled = False


def log(func):
    global logging_enabled, logger

    def wrapper(*args, **kwargs):
        global indentation_level
        if logging_enabled:
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

    return wrapper


def kill_loggers():
    global logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


# def log_function_execution(func):
#     """
#     Decorator function to log the name of the currently executing function or method if its module name starts with `climakitae`.
#     """
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         if logging_enabled:
#             # Get the module name of the function or method
#             module_name = func.__module__

#             # Check if the module name starts with 'climakitae'
#             if module_name.startswith('climakitae'):
#                 logging.info(f"Executing function or method: {func.__name__} in module: {module_name}")
#         return func(*args, **kwargs)

#     # Return the wrapped function or method
#     return wrapper

# def log_class_instantiation(cls):
#     """
#     Decorator function to log the instantiation of a class if its module name starts with `climakitae`.
#     """
#     class WrappedClass(cls):
#         def __new__(cls, *args, **kwargs):
#             if logging_enabled:
#                 # Get the module name of the class
#                 module_name = cls.__module__
#                 # Check if the module name starts with `climakitae`
#                 if module_name.startswith('climakitae'):
#                     logging.info(f"Instantiating class: {cls.__name__} in module: {module_name}")

#             # Create an instance of the class
#             instance = super().__new__(cls)

#             # Decorate functions/methods within the class
#             for name, obj in inspect.getmembers(instance):
#                 if inspect.isfunction(obj) or inspect.ismethod(obj):
#                     setattr(instance, name, log_function_execution(obj))

#             return instance

#     return WrappedClass

# def apply_logging_to_library_functions_and_methods():
#     """
#     Apply logging wrapper to all functions, methods, and classes within allowed modules.
#     """
#     for module_name in allowed_modules:
#         module = sys.modules[module_name]
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj) or inspect.ismethod(obj):
#                 setattr(module, name, log_function_execution(obj))
#             elif inspect.isclass(obj):
#                 pass
#                 # setattr(module, name, log_class_instantiation(obj))

# # Apply logging to all functions, methods, and classes within `climakitae`
# # apply_logging_to_library_functions_and_methods()


##### Writing out messages to stdout
# Create and configure logger
# import sys
# logger = logging.getLogger('Verbose Mode')
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(handler)

# # Example usage
# logger.debug('Debug message')
# logger.info('Info message')
# logger.warning('Warning message')
# logger.error('Error message')
# logger.critical('Critical message')

# for handler in logger.handlers[:]:
#     logger.removeHandler(handler)
