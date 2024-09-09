import pandas as pd


class Boundaries:
    """Get geospatial polygon data from the S3 stored parquet catalog.
    Used to access boundaries for subsetting data by state, county, etc.

    Attributes
    ----------
    _cat: intake.catalog.Catalog
        Parquet boundary catalog instance
    _us_states: pd.DataFrame
        Table of US state names and geometries
    _ca_counties: pd.DataFrame
        Table of California county names and geometries
        Sorted by county name alphabetical order
    _ca_watersheds: pd.DataFrame
        Table of California watershed names and geometries
        Sorted by watershed name alphabetical order
    _ca_utilities: pd.DataFrame
        Table of California IOUs and POUs, names and geometries
    _ca_forecast_zones: pd.DataFrame
        Table of California Demand Forecast Zones
    _ca_electric_balancing_areas: pd.DataFrame
        Table of Electric Balancing Areas

    Methods
    -------
    _get_us_states(self)
        Returns a dict of state abbreviations and indices
    _get_ca_counties(self)
        Returns a dict of California counties and their indices
    _get_ca_watersheds(self)
        Returns a dict for CA watersheds and their indices
    _get_forecast_zones(self)
        Returns a dict for CA electricity demand forecast zones
    _get_ious_pous(self)
        Returns a dict for CA electric load serving entities IOUs & POUs
    _get_electric_balancing_areas(self)
        Returns a dict for CA Electric Balancing Authority Areas
    """

    _cat = None
    _us_states = None
    _ca_counties = None
    _ca_watersheds = None
    _ca_utilities = None
    _ca_forecast_zones = None
    _ca_electric_balancing_areas = None

    def __init__(self, boundary_catalog):
        # Connect intake Catalog to class
        self._cat = boundary_catalog

    def load(self):
        """Read parquet files and sets class attributes."""
        self._us_states = self._cat.states.read()
        self._ca_counties = self._cat.counties.read().sort_values("NAME")
        self._ca_watersheds = self._cat.huc8.read().sort_values("Name")
        self._ca_utilities = self._cat.utilities.read()
        self._ca_forecast_zones = self._cat.dfz.read()
        self._ca_electric_balancing_areas = self._cat.eba.read()

        # EBA CALISO polygon has two options
        # One of the polygons is super tiny, with a negligible area
        # Perhaps this is an error from the producers of the data
        # Just grab the CALISO polygon with the large area
        tiny_caliso = self._ca_electric_balancing_areas.loc[
            (self._ca_electric_balancing_areas["NAME"] == "CALISO")
            & (self._ca_electric_balancing_areas["SHAPE_Area"] < 100)
        ].index
        self._ca_electric_balancing_areas = self._ca_electric_balancing_areas.drop(
            tiny_caliso
        )

        # For Forecast Zones named "Other", replace that with the name of the county
        self._ca_forecast_zones.loc[
            self._ca_forecast_zones["FZ_Name"] == "Other", "FZ_Name"
        ] = self._ca_forecast_zones["FZ_Def"]

    def _get_us_states(self):
        """
        Returns a custom sorted dictionary of western state abbreviations and indices.

        Returns
        -------
        dict

        """
        _states_subset_list = [
            "CA",
            "NV",
            "OR",
            "WA",
            "UT",
            "MT",
            "ID",
            "AZ",
            "CO",
            "NM",
            "WY",
        ]
        _us_states_subset = self._us_states.query("abbrevs in @_states_subset_list")[
            ["abbrevs"]
        ]
        _us_states_subset["abbrevs"] = pd.Categorical(
            _us_states_subset["abbrevs"], categories=_states_subset_list
        )
        _us_states_subset.sort_values(by="abbrevs", inplace=True)
        return dict(zip(_us_states_subset.abbrevs, _us_states_subset.index))

    def _get_ca_counties(self):
        """
        Returns a dictionary of California counties and their indices
        in the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_counties.index, index=self._ca_counties["NAME"]
        ).to_dict()

    def _get_ca_watersheds(self):
        """
        Returns a lookup dictionary for CA watersheds that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_watersheds.index, index=self._ca_watersheds["Name"]
        ).to_dict()

    def _get_forecast_zones(self):
        """
        Returns a lookup dictionary for CA Electricity Demand Forecast Zones that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_forecast_zones.index, index=self._ca_forecast_zones["FZ_Name"]
        ).to_dict()

    def _get_ious_pous(self):
        """
        Returns a lookup dictionary for CA Electric Load Serving Entities IOUs & POUs that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        put_at_top = [  # Put in the order you want it to appear in the dropdown
            "Pacific Gas & Electric Company",
            "San Diego Gas & Electric",
            "Southern California Edison",
            "Los Angeles Department of Water & Power",
            "Sacramento Municipal Utility District",
        ]
        other_IOUs_POUs_list = [
            ut for ut in self._ca_utilities["Utility"] if ut not in put_at_top
        ]
        other_IOUs_POUs_list = sorted(other_IOUs_POUs_list)  # Put in alphabetical order
        ordered_list = put_at_top + other_IOUs_POUs_list
        _subset = self._ca_utilities.query("Utility in @ordered_list")[["Utility"]]
        _subset["Utility"] = pd.Categorical(_subset["Utility"], categories=ordered_list)
        _subset.sort_values(by="Utility", inplace=True)
        return dict(zip(_subset["Utility"], _subset.index))

    def _get_electric_balancing_areas(self):
        """
        Returns a lookup dictionary for CA Electric Balancing Authority Areas that references
        the geoparquet file.

        Returns
        -------
        dict

        """
        return pd.Series(
            self._ca_electric_balancing_areas.index,
            index=self._ca_electric_balancing_areas["NAME"],
        ).to_dict()

    def boundary_dict(self):
        """Return a dict of the other boundary dicts, used to populate ck.Select.

        This returns a dictionary of lookup dictionaries for each set of
        geoparquet files that the user might be choosing from. It is used to
        populate the `DataParameters` cached_area dynamically as the category
        in the area_subset parameter changes.

        Returns
        -------
        dict

        """
        all_options = {
            "none": {"entire domain": 0},
            "lat/lon": {"coordinate selection": 0},
            "states": self._get_us_states(),
            "CA counties": self._get_ca_counties(),
            "CA watersheds": self._get_ca_watersheds(),
            "CA Electric Load Serving Entities (IOU & POU)": self._get_ious_pous(),
            "CA Electricity Demand Forecast Zones": self._get_forecast_zones(),
            "CA Electric Balancing Authority Areas": self._get_electric_balancing_areas(),
        }
        return all_options
