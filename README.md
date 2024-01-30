[![codecov](https://codecov.io/gh/laurap81/StarTracer/graph/badge.svg?token=ZHUR8G0A3Z)](https://codecov.io/gh/laurap81/StarTracer)

```
.                                        .   --   .                                       .
.                                    .                .                                   .
.                                  .                    .                                 .
*                                 .      StarTracer      .                                *
.                                 .                      .                                .
.                                  +                    .                                 .
.                                                     *                                   .
```


**StarTracer** is a library that allows to integrate star and cluster orbits with statistical sampling methods in order
to obtain an uncertainty estimation of the traceback result. It is mainly based on
[galpy](https://docs.galpy.org/en/v1.9.1/) which is an [astropy](https://www.astropy.org/index.html)
[affiliated packages](https://www.astropy.org/affiliated/). StarTracer is also based on astropy,
utilising the Table, SkyCoord, and Quantity classes.


### Features

The code offers two features: (i) integrating a cluster orbits, which is based on the average position and motion of
its cluster members and (ii) integrating orbits for individual stars (independent of cluster membership). 
Each method comes with different statistical sampling methods to estimate uncertainties for the orbit integration.

- For cluster orbit integration:
  - bootstrapping over the cluster members for an average cluster orbit integration
  - [] to be implemented: sampling from a fit to the 6D cluster distribution
  - methods to facilitate the calculation of averages and uncertainty estimation

- For stellar orbit integration:
  - Monte Carlo-type sampling from a normal distribution based on measurement and measurement uncertainties
  - again, methods to facilitate the calculation of averages and uncertainty estimation

- For representation:
  - Functions to visualise the resulting data (quick-check)


This is a versioned import package that can be installed locally. This project is partly still under construction and
will be updated accordingly.

### Installation

This package can be installed by cloning it and running


```
python3 -m pip install
```

from the top level directory, after which you can import the library to your code.


### How to use StarTracer

More examples can be found on the ReadTheDocs page of this project.
[!note: insert link]

1. loading in a data table and initialising a cluster based on this data:

```
>>> from StarTracer.startracer import Cluster
>>> from astropy.table import Table
>>> import numpy as np


>>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
>>> cluster = Cluster(cluster_data)
>>> cluster.data.head()

           ra        dec  ...  pmdec_error  radial_velocity_error
0  285.437166 -36.976161  ...     0.065070               0.081022
1  285.063548 -36.843633  ...     0.153469               0.318714
2  285.133199 -37.037393  ...     0.054875               0.183739
3  285.191013 -36.617811  ...     0.120342               0.696206
4  285.039615 -36.946535  ...     0.209726               0.046242
```

2. calculating orbits for 10 000 samples of the cluster over 10 Myr back in time for time steps of 0.1 Myr:

```
>>> sampled_cluster = cluster.sample_orbit(10, 0.1, number_of_samples=10000, direction='backward')
>>> integrated_orbits = sampled_cluster.get_data()
>>> print(f'type: {type(integrated_orbits)}, and shape of array: {np.shape(integrated_orbits)}')

type: <class 'numpy.ndarray'>, and shape of array: (7, 101, 10000)
```

This returns a class `SampledCluster` that stores the array with the resulting orbits. They can be accessed with
the `.get_data()` method and customarily post-processed.

However, this class also offers several methods to easily calculate the most commonly used average and uncertainties 
for the integrated orbits. See the [documentation]() for a full list of the available methods.

3. Each of the methods adds the calculated values as columns for each parameter to an attribute called
`.summary_dataframe` ([pandas.DataFrame](https://pandas.pydata.org/docs/reference/frame.html)).

```
>>> sampled_cluster.add_median()                    # the median of all sampled values for each
                                                    # position/ velocity per timestep
>>> sampled_cluster.add_mad()                       # the median absolut deviation
>>> sampled_cluster.add_percentile((2.5, 97.5))     # the 2.5 and 97.5 percentiles

>>> print(sampled_cluster.summary_dataframe.columns)

Index(['t', 'X_median', 'Y_median', 'Z_median', 'U_median', 'V_median', 'W_median',
       'X_mad', 'Y_mad', 'Z_mad', 'U_mad', 'V_mad', 'W_mad',
       'X_p2.5', 'Y_p2.5', 'Z_p2.5', 'U_p2.5', 'V_p2.5', 'W_p2.5', 'X_p97.5',
       'Y_p97.5', 'Z_p97.5', 'U_p97.5', 'V_p97.5', 'W_p97.5'],
      dtype='object')
```

4. After adding the intended averages and uncertainties they can easily be saved as \*.csv (default) or \*.fits files.

```
>>> sampled_cluster.save_table('Cluster_IntegratedOrbits.csv')
```

This works similarly for the `Stars` class that allows for orbit sampling from a normal distribution based on
measurement value and uncertainty.

