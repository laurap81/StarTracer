from startracer import Cluster, SampledCluster
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from astropy.coordinates import Galactocentric, LSR, Distance, SkyCoord
import astropy.units as unit
from astropy.table import Table

import pandas as pd
import warnings
import numpy as np

path_to_table = '../ExampleData/CrAChain_eRV1_stars_MC.csv'

itable = Table.read(path_to_table)
df = itable.to_pandas()

data = df[df['labels'] == 1]


CrA = Cluster(data)
CrA_sc = CrA.sample_orbit(5, 0.1)
CrA_sc.add_mean()
CrA_sc.add_std()

raw = CrA_sc.get_data()
a = 0