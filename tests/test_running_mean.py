

import climakitae as ck
from climakitae import timeseriestools as tst

_expected_rolling_avg = []

def test_running_mean():
    app = ck.Application()
    app.select(
        smoothing="running mean",
        timescale="monthly",
        timeslice = "2010-2020"
    )
    my_data = app.retrieve()
    my_data = my_data.compute()

    tsp = tst.TimeSeriesParams(my_data)

    tsp.transform_data() # transform_data calls _running_mean()
    assert tsp.output_current() == _expected_rolling_avg

