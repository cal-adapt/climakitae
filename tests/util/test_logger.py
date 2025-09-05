from _pytest.logging import LogCaptureFixture

from climakitae.util.logger import (
    current_logging_status,
    disable_app_logging,
    enable_app_logging,
    handler,
)
from climakitae.util.logger import logger as app_logger


class TestEnableAppLogging:

    def test_enable_app_logging(self, caplog: LogCaptureFixture):
        # Make sure logging is disabled at start
        disable_app_logging()

        # Test enabling logging
        enable_app_logging()
        assert len(caplog.records) == 0  # No logs should be recorded yet

    def test_enable_app_logging_with_message(self, caplog: LogCaptureFixture):
        # Make sure logging is disabled at start
        disable_app_logging()

        # Test enabling logging with a message
        enable_app_logging()
        app_logger.debug("This is a test log message.")
        assert len(caplog.records) == 1

    def test_enable_when_already_enabled(self):
        """Test that enabling logging when it's already enabled doesn't add duplicate handlers."""
        disable_app_logging()  # Start with clean state

        enable_app_logging()
        initial_handler_count = len(app_logger.handlers)

        # Try to enable again
        enable_app_logging()

        # Should not have added another handler
        assert len(app_logger.handlers) == initial_handler_count
        assert handler in app_logger.handlers


class TestDisableAppLogging:

    def test_disable_app_logging(self):
        """Test that disabling app logging prevents messages from being captured by logger handlers."""
        # First enable logging
        enable_app_logging()

        # Check if handler is attached to logger
        assert handler in app_logger.handlers

        # Then disable it
        disable_app_logging()

        # Check that handler is removed from logger
        assert handler not in app_logger.handlers

    def test_multiple_enable_disable_cycles(self):
        """Test that enabling and disabling logging multiple times works correctly."""
        # Start with disabled logging
        disable_app_logging()
        assert handler not in app_logger.handlers

        # First cycle
        enable_app_logging()
        assert handler in app_logger.handlers
        disable_app_logging()
        assert handler not in app_logger.handlers

        # Second cycle
        enable_app_logging()
        assert handler in app_logger.handlers
        disable_app_logging()
        assert handler not in app_logger.handlers


class TestCurrentLoggingStatus:

    def test_current_logging_status(self):
        # Initially should be disabled
        disable_app_logging()
        assert not current_logging_status()

        # After enabling, should be true
        enable_app_logging()
        assert current_logging_status()

        # After disabling again, should be false
        disable_app_logging()
        assert not current_logging_status()
