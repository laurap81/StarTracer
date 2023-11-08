import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

table = Table.read('../Data/CrAChain_eRV1_stars_MC.csv')
df = table.to_pandas()

df_cra = df[df['labels'] == 1]

CrA = SkyCoord(u=df_cra['x'].values * u.pc, v=df_cra['y'].values * u.pc, w=df_cra['z'].values * u.pc,
               U=df_cra['u'].values * u.km/u.s, V=df_cra['v'].values * u.km/u.s, W=df_cra['w'].values * u.km/u.s,
               frame='galactic',
               representation_type='cartesian',
               differential_type='cartesian')

cra_orbit = Orbit(CrA)

t = np.linspace(0, 10, 101) * u.Myr

cra_orbit.flip(inplace=True)
cra_orbit.integrate(t, pot=MWPotential2014)

print(np.shape(cra_orbit.vxvv))
print([np.round(x*1000) for x in cra_orbit.vxvv])
