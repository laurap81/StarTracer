import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt


def make_example_cluster(stars=50, save=False, make_plot=False):
    if stars < 1:
        raise ValueError('"stars" must be greater than 0.')

    ra = np.random.normal(285.4, 0.35, stars)
    dec = np.random.normal(-36.9, 0.15, stars)
    distance = np.random.normal(156.2, 3.7, stars)
    pmra = np.random.normal(4.9, 1., stars)
    pmdec = np.random.normal(-27.4, 0.9, stars)
    rv = np.random.normal(-1.6, 2.9, stars)

    ra_error = np.random.normal(0.08, 0.08, stars)
    dec_error = np.random.normal(0.08, 0.09, stars)
    distance_error = np.random.normal(2.5, 2.6, stars)
    pmra_error = np.random.normal(0.1, 0.1, stars)
    pmdec_error = np.random.normal(0.1, 0.1, stars)
    rv_error = np.random.normal(0.3, 0.2, stars)

    df = pd.DataFrame({'ra': ra, 'dec': dec, 'distance': distance, 'pmra': pmra, 'pmdec': pmdec, 'radial_velocity': rv,
                       'ra_error': ra_error, 'dec_error': dec_error, 'distance_error': distance_error,
                       'pmra_error': pmra_error, 'pmdec_error': pmdec_error, 'radial_velocity_error': rv_error})

    if save:
        df.to_csv('../ExampleData/ExampleCluster.csv')

        tbl = Table.from_pandas(df)
        tbl.write('../ExampleData/ExampleCluster.fits', format='fits', overwrite=True)
        print('Saved mock cluster as csv and fits at "../ExampleData/ExampleCluster.xxx".')

    if make_plot:
        f = plt.Figure(figsize=(6, 4.5))
        sx1 = f.add_subplot(221)
        sx1.plot(df.ra, df.dec, 'o', ms=8)
        sx1.set_xlim(np.min(df.ra) - .3 * (np.max(df.ra) - np.min(df.ra)),
                     np.max(df.ra) + .3 * (np.max(df.ra) - np.min(df.ra)))
        sx1.set_ylim(np.min(df.dec) - .3 * (np.max(df.dec) - np.min(df.dec)),
                     np.max(df.dec) + .3 * (np.max(df.dec) - np.min(df.dec)))
        sx1.set_xlabel('ra  --  deg')
        sx1.set_ylabel('dec  --  deg')

        sx2 = f.add_subplot(222)
        sx2.plot(df.pmra, df.pmdec, 'o', ms=8)
        sx2.set_xlim(np.min(df.pmra) - 1.5, np.max(df.pmra) + 1.5)
        sx2.set_ylim(np.min(df.pmdec) - 1.5, np.max(df.pmdec) + 1.5)
        sx2.set_xlabel('pmra  --  mas/yr')
        sx2.set_ylabel('pmdec  --  mas/yr')

        sx3 = f.add_subplot(223)
        sx3.hist(df.distance, bins=int(len(df) / (0.1 * len(df))))
        sx3.set_xlabel('distance  --  pc')
        sx3.set_ylabel('number of stars')

        sx4 = f.add_subplot(224)
        sx4.hist(df.radial_velocity, bins=int(len(df) / (0.1 * len(df))))
        sx4.set_xlabel('radial_velocity  --  km/s')
        sx4.set_ylabel('number of stars')

        f.tight_layout()
        f.savefig("../ExampleData/ExampleCluster.jpg", dpi=200, format='jpg')
        print('Plotting...')


if __name__ == "__main__":
    make_example_cluster(stars=50, save=True, make_plot=True)
