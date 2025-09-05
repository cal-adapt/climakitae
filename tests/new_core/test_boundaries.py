"""Unit tests for the new_core boundaries module.

This module provides comprehensive test coverage for the Boundaries class,
including lazy loading behavior, catalog validation, data processing,
memory management, and error handling.

"""

import warnings
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from climakitae.core.constants import WESTERN_STATES_LIST
from climakitae.new_core.data_access.boundaries import Boundaries


class TestBoundariesInitialization:
    """Test Boundaries class initialization and validation."""

    def test_init_with_valid_catalog(self):
        """Test initialization with a valid catalog."""
        mock_catalog = Mock()
        # Add required attributes
        for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
            setattr(mock_catalog, attr, Mock())

        boundaries = Boundaries(mock_catalog)
        assert boundaries._cat is mock_catalog
        assert boundaries._lookup_cache == {}
        # Verify all private DataFrames are None initially
        # Test that lazy loading hasn't occurred yet by checking properties don't load data
        # We'll verify this by testing the _lookup_cache is empty initially
        assert boundaries._lookup_cache == {}

    def test_init_with_missing_catalog_entries(self):
        """Test initialization fails with missing catalog entries."""
        mock_catalog = Mock(spec=[])  # Empty spec means no attributes
        # Only add some required attributes
        setattr(mock_catalog, "states", Mock())
        setattr(mock_catalog, "counties", Mock())
        # Missing: huc8, utilities, dfz, eba

        with pytest.raises(ValueError) as excinfo:
            Boundaries(mock_catalog)

        assert "Missing required catalog entries" in str(excinfo.value)
        assert "huc8" in str(excinfo.value)
        assert "utilities" in str(excinfo.value)
        assert "dfz" in str(excinfo.value)
        assert "eba" in str(excinfo.value)

    def test_validate_catalog_success(self):
        """Test successful catalog validation."""
        mock_catalog = Mock()
        for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
            setattr(mock_catalog, attr, Mock())

        boundaries = Boundaries(mock_catalog)
        # Should not raise any exception
        boundaries.validate_catalog()

    def test_validate_catalog_failure(self):
        """Test catalog validation failure."""
        mock_catalog = Mock(spec=[])  # Empty spec means no attributes
        setattr(mock_catalog, "states", Mock())
        # Missing other required attributes

        boundaries = Boundaries.__new__(Boundaries)  # Create without __init__
        boundaries._cat = mock_catalog

        with pytest.raises(ValueError) as excinfo:
            boundaries.validate_catalog()

        assert "Missing required catalog entries" in str(excinfo.value)


class TestBoundariesProperties:
    """Test lazy loading properties and data processing."""

    @pytest.fixture
    def mock_boundaries(self):
        """Create a Boundaries instance with mocked catalog."""
        mock_catalog = Mock()
        for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
            catalog_entry = Mock()
            catalog_entry.read.return_value = self._create_mock_dataframe(attr)
            setattr(mock_catalog, attr, catalog_entry)

        return Boundaries(mock_catalog)

    def _create_mock_dataframe(self, dataset_type):
        """Create mock DataFrames with appropriate structure for each dataset type."""
        if dataset_type == "states":
            return pd.DataFrame(
                {
                    "abbrevs": [
                        "CA",
                        "NV",
                        "OR",
                        "WA",
                        "NY",
                    ],  # Include non-western state
                    "name": [
                        "California",
                        "Nevada",
                        "Oregon",
                        "Washington",
                        "New York",
                    ],
                    "geometry": [f"POLYGON_{i}" for i in range(5)],
                }
            )
        elif dataset_type == "counties":
            return pd.DataFrame(
                {
                    "NAME": ["Los Angeles", "San Francisco", "Alameda", "San Diego"],
                    "geometry": [f"POLYGON_{i}" for i in range(4)],
                }
            )
        elif dataset_type == "huc8":
            return pd.DataFrame(
                {
                    "Name": ["Sacramento River", "San Francisco Bay", "Central Valley"],
                    "geometry": [f"POLYGON_{i}" for i in range(3)],
                }
            )
        elif dataset_type == "utilities":
            return pd.DataFrame(
                {
                    "Utility": [
                        "Pacific Gas & Electric Company",
                        "San Diego Gas & Electric",
                        "Other Utility",
                    ],
                    "geometry": [f"POLYGON_{i}" for i in range(3)],
                }
            )
        elif dataset_type == "dfz":
            return pd.DataFrame(
                {
                    "FZ_Name": ["North Bay", "Other", "South Bay"],
                    "FZ_Def": ["North Bay", "Alameda County", "South Bay"],
                    "geometry": [f"POLYGON_{i}" for i in range(3)],
                }
            )
        elif dataset_type == "eba":
            return pd.DataFrame(
                {
                    "NAME": ["CALISO", "CALISO", "Other EBA"],
                    "SHAPE_Area": [
                        50,
                        150,
                        100,
                    ],  # First CALISO is tiny, second is large
                    "geometry": [f"POLYGON_{i}" for i in range(3)],
                }
            )

    def test_us_states_lazy_loading(self, mock_boundaries):
        """Test lazy loading of US states data."""
        # Initially not loaded - check private attribute
        assert getattr(mock_boundaries, "_Boundaries__us_states", None) is None

        # Access triggers loading
        states = mock_boundaries._us_states
        assert states is not None
        assert isinstance(states, pd.DataFrame)
        assert getattr(mock_boundaries, "_Boundaries__us_states", None) is not None

        # Subsequent access uses cached data
        states2 = mock_boundaries._us_states
        assert states is states2

    def test_us_states_loading_error(self):
        """Test error handling during US states loading."""
        mock_catalog = Mock()
        catalog_entry = Mock()
        catalog_entry.read.side_effect = Exception("Catalog read error")
        setattr(mock_catalog, "states", catalog_entry)

        # Set other required attributes
        for attr in ["counties", "huc8", "utilities", "dfz", "eba"]:
            setattr(mock_catalog, attr, Mock())

        boundaries = Boundaries(mock_catalog)

        with pytest.raises(RuntimeError) as excinfo:
            _ = boundaries._us_states

        assert "Failed to load US states data" in str(excinfo.value)

    def test_ca_counties_lazy_loading(self, mock_boundaries):
        """Test lazy loading of CA counties data."""
        # Initially not loaded
        assert getattr(mock_boundaries, "_Boundaries__ca_counties", None) is None

        # Access triggers loading
        counties = mock_boundaries._ca_counties
        assert counties is not None
        assert isinstance(counties, pd.DataFrame)
        assert getattr(mock_boundaries, "_Boundaries__ca_counties", None) is not None

    def test_ca_counties_loading_error(self):
        """Test error handling during CA counties loading."""
        mock_catalog = Mock()
        catalog_entry = Mock()
        catalog_entry.read.side_effect = Exception("Catalog read error")
        setattr(mock_catalog, "counties", catalog_entry)

        # Set other required attributes
        for attr in ["states", "huc8", "utilities", "dfz", "eba"]:
            setattr(mock_catalog, attr, Mock())

        boundaries = Boundaries(mock_catalog)

        with pytest.raises(RuntimeError) as excinfo:
            _ = boundaries._ca_counties

        assert "Failed to load CA counties data" in str(excinfo.value)

    def test_ca_watersheds_lazy_loading(self, mock_boundaries):
        """Test lazy loading of CA watersheds data."""
        # Initially not loaded
        assert mock_boundaries._Boundaries__ca_watersheds is None

        # Access triggers loading
        watersheds = mock_boundaries._ca_watersheds
        assert watersheds is not None
        assert isinstance(watersheds, pd.DataFrame)

    def test_ca_utilities_lazy_loading(self, mock_boundaries):
        """Test lazy loading of CA utilities data."""
        utilities = mock_boundaries._ca_utilities
        assert utilities is not None
        assert isinstance(utilities, pd.DataFrame)

    def test_ca_forecast_zones_lazy_loading(self, mock_boundaries):
        """Test lazy loading of CA forecast zones data."""
        zones = mock_boundaries._ca_forecast_zones
        assert zones is not None
        assert isinstance(zones, pd.DataFrame)

    def test_ca_electric_balancing_areas_lazy_loading(self, mock_boundaries):
        """Test lazy loading of CA electric balancing areas data."""
        areas = mock_boundaries._ca_electric_balancing_areas
        assert areas is not None
        assert isinstance(areas, pd.DataFrame)

    def test_property_setters(self, mock_boundaries):
        """Test property setters work correctly."""
        test_df = pd.DataFrame({"test": [1, 2, 3]})

        mock_boundaries._us_states = test_df
        assert getattr(mock_boundaries, "_Boundaries__us_states", None) is test_df

        mock_boundaries._ca_counties = test_df
        assert getattr(mock_boundaries, "_Boundaries__ca_counties", None) is test_df


class TestBoundariesDataProcessing:
    """Test data processing methods."""

    def test_process_us_states(self):
        """Test US states data processing."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame(
            {"abbrevs": ["CA", "OR"], "name": ["California", "Oregon"]}
        )

        result = boundaries._process_us_states(test_df)

        # Currently no processing, should return same DataFrame
        pd.testing.assert_frame_equal(result, test_df)

    def test_process_ca_counties(self):
        """Test CA counties data processing (sorting)."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame({"NAME": ["San Francisco", "Alameda", "Los Angeles"]})

        result = boundaries._process_ca_counties(test_df)

        # Should be sorted by NAME
        expected = test_df.sort_values("NAME")
        pd.testing.assert_frame_equal(result, expected)

    def test_process_ca_watersheds(self):
        """Test CA watersheds data processing (sorting)."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame({"Name": ["Sacramento", "Bay Area", "Central Valley"]})

        result = boundaries._process_ca_watersheds(test_df)

        # Should be sorted by Name
        expected = test_df.sort_values("Name")
        pd.testing.assert_frame_equal(result, expected)

    def test_process_ca_utilities(self):
        """Test CA utilities data processing."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame({"Utility": ["PG&E", "SCE"]})

        result = boundaries._process_ca_utilities(test_df)

        # Currently no processing, should return same DataFrame
        pd.testing.assert_frame_equal(result, test_df)

    def test_process_ca_forecast_zones(self):
        """Test CA forecast zones data processing (replace 'Other' names)."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame(
            {
                "FZ_Name": ["North Bay", "Other", "South Bay"],
                "FZ_Def": ["North Bay", "Alameda County", "South Bay"],
            }
        )

        result = boundaries._process_ca_forecast_zones(test_df)

        # 'Other' in FZ_Name should be replaced with FZ_Def value
        assert result.loc[1, "FZ_Name"] == "Alameda County"
        assert result.loc[0, "FZ_Name"] == "North Bay"  # Unchanged
        assert result.loc[2, "FZ_Name"] == "South Bay"  # Unchanged

    def test_process_ca_electric_balancing_areas(self):
        """Test CA electric balancing areas processing (remove tiny CALISO)."""
        boundaries = Boundaries.__new__(Boundaries)
        test_df = pd.DataFrame(
            {
                "NAME": ["CALISO", "CALISO", "Other EBA"],
                "SHAPE_Area": [50, 150, 100],  # First CALISO is tiny (< 100)
            },
            index=[0, 1, 2],
        )

        result = boundaries._process_ca_electric_balancing_areas(test_df)

        # Tiny CALISO should be removed
        assert len(result) == 2
        assert 0 not in result.index  # Tiny CALISO removed
        assert 1 in result.index  # Large CALISO kept
        assert 2 in result.index  # Other EBA kept


class TestBoundariesLookupMethods:
    """Test lookup dictionary building and caching."""

    @pytest.fixture
    def mock_boundaries_with_data(self):
        """Create boundaries with mock data already loaded."""
        boundaries = Boundaries.__new__(Boundaries)
        boundaries._lookup_cache = {}

        # Mock data for different boundary types using setattr
        setattr(
            boundaries,
            "_Boundaries__us_states",
            pd.DataFrame(
                {
                    "abbrevs": ["CA", "OR", "WA", "NV"],  # Some western states
                    "name": ["California", "Oregon", "Washington", "Nevada"],
                },
                index=[10, 11, 12, 13],
            ),
        )

        setattr(
            boundaries,
            "_Boundaries__ca_counties",
            pd.DataFrame(
                {"NAME": ["Alameda", "Los Angeles", "San Francisco"]},
                index=[20, 21, 22],
            ),
        )

        setattr(
            boundaries,
            "_Boundaries__ca_watersheds",
            pd.DataFrame(
                {"Name": ["Central Valley", "San Francisco Bay"]}, index=[30, 31]
            ),
        )

        setattr(
            boundaries,
            "_Boundaries__ca_utilities",
            pd.DataFrame(
                {
                    "Utility": [
                        "Pacific Gas & Electric Company",
                        "Other Utility",
                        "San Diego Gas & Electric",
                    ]
                },
                index=[40, 41, 42],
            ),
        )

        setattr(
            boundaries,
            "_Boundaries__ca_forecast_zones",
            pd.DataFrame({"FZ_Name": ["North Bay", "South Bay"]}, index=[50, 51]),
        )

        setattr(
            boundaries,
            "_Boundaries__ca_electric_balancing_areas",
            pd.DataFrame({"NAME": ["CALISO", "Other EBA"]}, index=[60, 61]),
        )

        return boundaries

    def test_get_us_states_caching(self, mock_boundaries_with_data):
        """Test US states lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        # First call builds cache
        result1 = boundaries._get_us_states()
        assert "us_states" in boundaries._lookup_cache
        assert isinstance(result1, dict)

        # Second call uses cache
        result2 = boundaries._get_us_states()
        assert result1 is result2

    def test_build_us_states_lookup(self, mock_boundaries_with_data):
        """Test building US states lookup with western states ordering."""
        boundaries = mock_boundaries_with_data

        with patch("pandas.Categorical") as mock_categorical:
            us_states_df = getattr(boundaries, "_Boundaries__us_states")
            mock_categorical.return_value = us_states_df["abbrevs"]
            result = boundaries._build_us_states_lookup()

        assert isinstance(result, dict)
        # Should only include western states
        for state in result.keys():
            assert state in WESTERN_STATES_LIST

    def test_get_ca_counties_caching(self, mock_boundaries_with_data):
        """Test CA counties lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        result1 = boundaries._get_ca_counties()
        assert "ca_counties" in boundaries._lookup_cache

        result2 = boundaries._get_ca_counties()
        assert result1 is result2

    def test_get_ca_watersheds_caching(self, mock_boundaries_with_data):
        """Test CA watersheds lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        result = boundaries._get_ca_watersheds()
        assert "ca_watersheds" in boundaries._lookup_cache
        assert isinstance(result, dict)
        assert "Central Valley" in result
        assert "San Francisco Bay" in result

    def test_get_forecast_zones_caching(self, mock_boundaries_with_data):
        """Test forecast zones lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        result = boundaries._get_forecast_zones()
        assert "forecast_zones" in boundaries._lookup_cache
        assert "North Bay" in result
        assert "South Bay" in result

    def test_get_ious_pous_caching(self, mock_boundaries_with_data):
        """Test IOUs/POUs lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        result1 = boundaries._get_ious_pous()
        assert "ious_pous" in boundaries._lookup_cache

        result2 = boundaries._get_ious_pous()
        assert result1 is result2

    def test_build_ious_pous_lookup_priority_ordering(self, mock_boundaries_with_data):
        """Test IOUs/POUs lookup respects priority ordering."""
        boundaries = mock_boundaries_with_data

        with patch("pandas.Categorical") as mock_categorical:
            ca_utilities_df = getattr(boundaries, "_Boundaries__ca_utilities")
            mock_categorical.return_value = ca_utilities_df["Utility"]
            result = boundaries._build_ious_pous_lookup()

        # Priority utilities should be included
        assert "Pacific Gas & Electric Company" in result
        assert "San Diego Gas & Electric" in result
        # Non-priority utility should also be included
        assert "Other Utility" in result

    def test_get_electric_balancing_areas_caching(self, mock_boundaries_with_data):
        """Test electric balancing areas lookup dictionary caching."""
        boundaries = mock_boundaries_with_data

        result = boundaries._get_electric_balancing_areas()
        assert "electric_balancing_areas" in boundaries._lookup_cache
        assert "CALISO" in result
        assert "Other EBA" in result


class TestBoundariesPublicMethods:
    """Test public API methods."""

    @pytest.fixture
    def mock_boundaries_public(self):
        """Create boundaries for testing public methods."""
        mock_catalog = Mock()
        for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
            catalog_entry = Mock()
            catalog_entry.read.return_value = pd.DataFrame({"test": [1, 2, 3]})
            setattr(mock_catalog, attr, catalog_entry)

        boundaries = Boundaries(mock_catalog)

        # Mock all getter methods to avoid complex setup
        boundaries._get_us_states = Mock(return_value={"CA": 0, "OR": 1})
        boundaries._get_ca_counties = Mock(
            return_value={"Alameda": 0, "Los Angeles": 1}
        )
        boundaries._get_ca_watersheds = Mock(return_value={"Central Valley": 0})
        boundaries._get_ious_pous = Mock(return_value={"PG&E": 0})
        boundaries._get_forecast_zones = Mock(return_value={"North Bay": 0})
        boundaries._get_electric_balancing_areas = Mock(return_value={"CALISO": 0})

        return boundaries

    def test_boundary_dict(self, mock_boundaries_public):
        """Test boundary_dict returns complete dictionary structure."""
        result = mock_boundaries_public.boundary_dict()

        expected_keys = [
            "none",
            "lat/lon",
            "states",
            "CA counties",
            "CA watersheds",
            "CA Electric Load Serving Entities (IOU & POU)",
            "CA Electricity Demand Forecast Zones",
            "CA Electric Balancing Authority Areas",
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], dict)

        # Check that getter methods were called
        mock_boundaries_public._get_us_states.assert_called_once()
        mock_boundaries_public._get_ca_counties.assert_called_once()
        mock_boundaries_public._get_ca_watersheds.assert_called_once()
        mock_boundaries_public._get_ious_pous.assert_called_once()
        mock_boundaries_public._get_forecast_zones.assert_called_once()
        mock_boundaries_public._get_electric_balancing_areas.assert_called_once()

    def test_load_deprecated_warning(self, mock_boundaries_public):
        """Test that load() method issues deprecation warning."""
        with patch.object(mock_boundaries_public, "preload_all") as mock_preload:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mock_boundaries_public.load()

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()
                mock_preload.assert_called_once()

    def test_preload_all(self, mock_boundaries_public):
        """Test preload_all forces loading of all data."""
        # Mock the actual property accesses to avoid complex nested patches
        mock_boundaries_public._us_states = Mock()
        mock_boundaries_public._ca_counties = Mock()
        mock_boundaries_public._ca_watersheds = Mock()
        mock_boundaries_public._ca_utilities = Mock()
        mock_boundaries_public._ca_forecast_zones = Mock()
        mock_boundaries_public._ca_electric_balancing_areas = Mock()

        mock_boundaries_public.preload_all()

        # Verify all getter methods were called for cache building
        mock_boundaries_public._get_us_states.assert_called()
        mock_boundaries_public._get_ca_counties.assert_called()

    def test_clear_cache(self, mock_boundaries_public):
        """Test clear_cache resets all cached data."""
        # Set some data first
        mock_boundaries_public._lookup_cache["test"] = {"key": "value"}
        setattr(mock_boundaries_public, "_Boundaries__us_states", Mock())
        setattr(mock_boundaries_public, "_Boundaries__ca_counties", Mock())

        mock_boundaries_public.clear_cache()

        # All should be cleared
        assert mock_boundaries_public._lookup_cache == {}
        # Check key attributes are None after clearing
        private_attrs = [
            "_Boundaries__us_states",
            "_Boundaries__ca_counties",
            "_Boundaries__ca_watersheds",
            "_Boundaries__ca_utilities",
            "_Boundaries__ca_forecast_zones",
            "_Boundaries__ca_electric_balancing_areas",
        ]
        for attr in private_attrs:
            assert getattr(mock_boundaries_public, attr, None) is None


class TestBoundariesMemoryManagement:
    """Test memory usage monitoring and management."""

    def test_get_memory_usage_no_data_loaded(self):
        """Test memory usage when no data is loaded."""
        boundaries = Boundaries.__new__(Boundaries)
        boundaries._lookup_cache = {}

        # Set all private DataFrames to None (not loaded) using setattr
        setattr(boundaries, "_Boundaries__us_states", None)
        setattr(boundaries, "_Boundaries__ca_counties", None)
        setattr(boundaries, "_Boundaries__ca_watersheds", None)
        setattr(boundaries, "_Boundaries__ca_utilities", None)
        setattr(boundaries, "_Boundaries__ca_forecast_zones", None)
        setattr(boundaries, "_Boundaries__ca_electric_balancing_areas", None)

        result = boundaries.get_memory_usage()

        # All dataset usage should be 0
        assert result["us_states"] == 0
        assert result["ca_counties"] == 0
        assert result["ca_watersheds"] == 0
        assert result["ca_utilities"] == 0
        assert result["ca_forecast_zones"] == 0
        assert result["ca_electric_balancing_areas"] == 0
        assert result["total_bytes"] == 0
        assert result["loaded_datasets"] == 0
        assert result["cached_lookups"] == 0

    def test_get_memory_usage_with_loaded_data(self):
        """Test memory usage with some data loaded."""
        boundaries = Boundaries.__new__(Boundaries)
        boundaries._lookup_cache = {"test": {"key": 0}}

        # Mock DataFrame with memory usage
        mock_df1 = Mock()
        mock_memory_usage = Mock()
        mock_memory_usage.sum.return_value = 1024
        mock_df1.memory_usage.return_value = mock_memory_usage

        mock_df2 = Mock()
        mock_memory_usage2 = Mock()
        mock_memory_usage2.sum.return_value = 2048
        mock_df2.memory_usage.return_value = mock_memory_usage2

        # Set some DataFrames as loaded, others as None using setattr
        setattr(boundaries, "_Boundaries__us_states", mock_df1)
        setattr(boundaries, "_Boundaries__ca_counties", mock_df2)
        setattr(boundaries, "_Boundaries__ca_watersheds", None)
        setattr(boundaries, "_Boundaries__ca_utilities", None)
        setattr(boundaries, "_Boundaries__ca_forecast_zones", None)
        setattr(boundaries, "_Boundaries__ca_electric_balancing_areas", None)

        result = boundaries.get_memory_usage()

        assert result["us_states"] == 1024
        assert result["ca_counties"] == 2048
        assert result["ca_watersheds"] == 0
        assert result["total_bytes"] == 3072
        assert result["loaded_datasets"] == 2
        assert result["cached_lookups"] == 1
        assert result["total_human"] == "3.0 KB"

    def test_format_bytes(self):
        """Test byte formatting utility."""
        assert Boundaries._format_bytes(512) == "512.0 B"
        assert Boundaries._format_bytes(1024) == "1.0 KB"
        assert Boundaries._format_bytes(1536) == "1.5 KB"
        assert Boundaries._format_bytes(1024 * 1024) == "1.0 MB"
        assert Boundaries._format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert Boundaries._format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestBoundariesEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframes(self):
        """Test handling of empty DataFrames."""
        boundaries = Boundaries.__new__(Boundaries)

        # Test with empty DataFrame that has the required column
        empty_df = pd.DataFrame(columns=["NAME"])
        result = boundaries._process_ca_counties(empty_df)
        assert len(result) == 0
        assert "NAME" in result.columns

        # Test with completely empty DataFrame
        completely_empty_df = pd.DataFrame()
        with pytest.raises(KeyError):
            # This should raise KeyError since 'NAME' column doesn't exist
            boundaries._process_ca_counties(completely_empty_df)

    def test_dataframe_with_no_matching_states(self):
        """Test handling when no western states are found."""
        boundaries = Boundaries.__new__(Boundaries)
        setattr(
            boundaries,
            "_Boundaries__us_states",
            pd.DataFrame(
                {
                    "abbrevs": ["NY", "FL"],  # No western states
                    "name": ["New York", "Florida"],
                }
            ),
        )
        boundaries._lookup_cache = {}

        result = boundaries._build_us_states_lookup()
        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

    def test_forecast_zones_with_no_other_entries(self):
        """Test forecast zones processing when no 'Other' entries exist."""
        boundaries = Boundaries.__new__(Boundaries)

        test_df = pd.DataFrame(
            {
                "FZ_Name": ["North Bay", "South Bay"],
                "FZ_Def": ["North Bay", "South Bay"],
            }
        )

        result = boundaries._process_ca_forecast_zones(test_df)

        # Should remain unchanged
        pd.testing.assert_frame_equal(result, test_df)

    def test_electric_balancing_areas_with_no_tiny_caliso(self):
        """Test electric balancing areas when no tiny CALISO exists."""
        boundaries = Boundaries.__new__(Boundaries)

        test_df = pd.DataFrame(
            {
                "NAME": ["CALISO", "Other EBA"],
                "SHAPE_Area": [150, 100],  # No tiny CALISO
            }
        )

        result = boundaries._process_ca_electric_balancing_areas(test_df)

        # Should remain unchanged
        pd.testing.assert_frame_equal(result, test_df)

    def test_utilities_with_no_priority_utilities(self):
        """Test utilities processing when no priority utilities exist."""
        boundaries = Boundaries.__new__(Boundaries)
        setattr(
            boundaries,
            "_Boundaries__ca_utilities",
            pd.DataFrame({"Utility": ["Random Utility A", "Random Utility B"]}),
        )
        boundaries._lookup_cache = {}

        with patch("pandas.Categorical") as mock_categorical:
            ca_utilities_df = getattr(boundaries, "_Boundaries__ca_utilities")
            mock_categorical.return_value = ca_utilities_df["Utility"]
            result = boundaries._build_ious_pous_lookup()

        assert isinstance(result, dict)
        assert "Random Utility A" in result
        assert "Random Utility B" in result


if __name__ == "__main__":
    pytest.main([__file__])
