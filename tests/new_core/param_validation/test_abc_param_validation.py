"""
Unit tests for climakitae/new_core/param_validation/abc_param_validation.py

This module contains comprehensive unit tests for the parameter validation
framework including decorator registration, abstract base class functionality,
and validation logic.
"""

from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY,
    register_catalog_validator,
)


class TestRegisterCatalogValidator:
    """Test class for the register_catalog_validator decorator."""

    def setup_method(self):
        """Clear the registry before each test."""
        _CATALOG_VALIDATOR_REGISTRY.clear()

    def test_register_catalog_validator_successful(self):
        """Test successful registration of a catalog validator."""

        @register_catalog_validator("test_catalog")
        class TestValidator:
            pass

        assert "test_catalog" in _CATALOG_VALIDATOR_REGISTRY
        assert _CATALOG_VALIDATOR_REGISTRY["test_catalog"] is TestValidator

    def test_register_catalog_validator_returns_class(self):
        """Test that decorator returns the class unchanged."""

        @register_catalog_validator("test_catalog")
        class TestValidator:
            def test_method(self):
                return "test"

        # Class should be returned unchanged
        instance = TestValidator()
        assert instance.test_method() == "test"

    def test_register_multiple_validators(self):
        """Test registering multiple catalog validators."""

        @register_catalog_validator("catalog1")
        class Validator1:
            pass

        @register_catalog_validator("catalog2")
        class Validator2:
            pass

        assert "catalog1" in _CATALOG_VALIDATOR_REGISTRY
        assert "catalog2" in _CATALOG_VALIDATOR_REGISTRY
        assert _CATALOG_VALIDATOR_REGISTRY["catalog1"] is Validator1
        assert _CATALOG_VALIDATOR_REGISTRY["catalog2"] is Validator2

    def test_register_overwrite_existing(self):
        """Test that registering with same name overwrites existing."""

        @register_catalog_validator("test_catalog")
        class Validator1:
            pass

        @register_catalog_validator("test_catalog")
        class Validator2:
            pass

        assert _CATALOG_VALIDATOR_REGISTRY["test_catalog"] is Validator2
