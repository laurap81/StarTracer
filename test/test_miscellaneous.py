from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import StarTracer.startracer as st

filepath = './StarTracer/example_data/ExampleCluster_1.fits'


def test_skycoord():
    SC = st.skycoord_from_table(filepath)
    assert isinstance(SC, SkyCoord)
