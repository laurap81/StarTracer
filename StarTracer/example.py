from startracer import Cluster, SampledCluster, Stars
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from astropy.coordinates import Galactocentric, LSR, Distance, SkyCoord
import astropy.units as unit
from astropy.table import Table

import pandas as pd
import warnings
import numpy as np

path_to_table = 'example_data/CrAChain_eRV1_stars_MC.csv'

itable = Table.read(path_to_table)
df = itable.to_pandas()

data = df[df['labels'] == 1]


# CrA = Cluster(data)
# CrA_sc = CrA.sample_orbit(5, 0.1)
# CrA_sc.add_mean()
# CrA_sc.add_std()
#
# raw = CrA_sc.get_data()
time_end = 5
time_step = 0.1
number_of_samples = 100
# sampled_stars_orbits = test_stars.sample_orbit(time_end=time_end, time_step=time_step,
#                                                number_of_samples=number_of_samples_, direction='backward')
CrA = Stars(data)
CrA_orbits = CrA.sample_orbit(time_end, time_step, number_of_samples=number_of_samples, direction='backward')
CrA_orbits.get_data()
a = 0