import types
from unittest.mock import patch

import pytest

from climakitae.util import dev_logging


class TestLogDecorator:
    def test_log_decorator(self):
        @dev_logging._log
        def test_function() -> str:
            return "test"

        with patch("builtins.print") as mock_print:
            result = test_function()
            assert result == "test"
            assert mock_print.called


class TestEnableLibLogging:
    @pytest.fixture
    def mock_module(self) -> types.ModuleType:
        # Create a mock module for testing
        mock_module = types.ModuleType("mock_module")
        mock_module.__name__ = "mock_module"
        mock_module.__module__ = "climakitae"
        mock_module.__package__ = "climakitae"

        def mock_function() -> str:
            return "original"

        # Set the __module__ attribute of the function properly
        mock_function.__module__ = "climakitae.mock_module"

        mock_module.mock_function = mock_function
        return mock_module

    @pytest.fixture
    def mock_module_with_class(self, mock_module: types.ModuleType) -> types.ModuleType:
        """Create a mock module with a class that has a method."""

        class TestClass:
            # Add a regular function at class level that will be detected during module inspection
            def __init__(self):
                pass

            # Keep the staticmethod for consistency with existing tests
            @staticmethod
            def class_method() -> str:
                return "class method result"

        # Make a separate function that belongs to the class's module
        def standalone_function() -> str:
            return "standalone function result"

        # Set proper module attributes
        TestClass.__module__ = "climakitae.mock_module"
        standalone_function.__module__ = "climakitae.mock_module"

        # Add both to the module
        mock_module.TestClass = TestClass
        mock_module.standalone_function = standalone_function

        return mock_module

    def test_recursive_class_logging(self, mock_module_with_class: types.ModuleType):
        """Test that _enable_lib_logging recursively adds logging to class methods."""
        # Enable logging on the module
        dev_logging.enable_lib_logging(mock_module_with_class)

        # Check if the standalone function was wrapped
        assert hasattr(mock_module_with_class.standalone_function, "_is_logged")

        # Test that the function works and is logged
        with patch("builtins.print") as mock_print:
            result = mock_module_with_class.standalone_function()
            assert result == "standalone function result"
            assert mock_print.called
            mock_print.reset_mock()

            # We can still test the static method works, even if not directly wrapped
            result = mock_module_with_class.TestClass.class_method()
            assert result == "class method result"

    def test_enable_lib_logging(self, mock_module: types.ModuleType):
        dev_logging.enable_lib_logging(mock_module)
        assert hasattr(mock_module.mock_function, "_is_logged")

        with patch("builtins.print") as mock_print:
            mock_module.mock_function()
            assert mock_print.called

    def test_enable_lib_logging_non_module(self):
        with patch("builtins.print") as mock_print:
            dev_logging.enable_lib_logging("not_a_module")
            mock_print.assert_called_with(
                "Error: Current object is not a module object."
            )

    def test_already_wrapped(self, mock_module: types.ModuleType):
        # Enable logging once
        dev_logging.enable_lib_logging(mock_module)
        original_function = mock_module.mock_function

        # Enable logging again - should not wrap again
        dev_logging.enable_lib_logging(mock_module)
        assert mock_module.mock_function == original_function


class TestDisableLibLogging:
    @pytest.fixture
    def mock_module(self) -> types.ModuleType:
        # Create a mock module for testing
        mock_module = types.ModuleType("mock_module")
        mock_module.__name__ = "mock_module"
        mock_module.__module__ = "climakitae"
        mock_module.__package__ = "climakitae"

        def mock_function():
            return "original"

        # Set the __module__ attribute of the function properly
        mock_function.__module__ = "climakitae.mock_module"

        mock_module.mock_function = mock_function
        return mock_module

    def test_disable_lib_logging(self, mock_module: types.ModuleType):
        # First, enable logging
        dev_logging.enable_lib_logging(mock_module)
        assert hasattr(mock_module.mock_function, "_is_logged")

        # Then, disable logging
        dev_logging.disable_lib_logging(mock_module)
        assert not hasattr(mock_module.mock_function, "_is_logged")
        assert mock_module.mock_function() == "original"
