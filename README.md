[![codecov](https://codecov.io/gh/laurap81/StarTracer/graph/badge.svg?token=ZHUR8G0A3Z)](https://codecov.io/gh/laurap81/StarTracer)


# StarTracer
```
  .                                        .   --   .                                       .
  .                                    .                .                                   .
  .                                  .                    .                                 .
  *                                 .      StarTracer      .                                *
  .                                 .                      .                                .
  .                                  +                    .                                 .
  .                                                     *                                   .
```


**StarTracer** is a python package that allows to integrate star and cluster orbits with statistical sampling methods
in order to obtain an uncertainty estimation of the traceback result. It is mainly based on
[galpy](https://docs.galpy.org/en/v1.9.1/) which is an [astropy](https://www.astropy.org/index.html)
[affiliated package](https://www.astropy.org/affiliated/). StarTracer is also based on astropy,
utilising the [Table](https://docs.astropy.org/en/stable/api/astropy.table.Table.html),
[SkyCoord](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html),
and [Quantity](https://docs.astropy.org/en/stable/units/quantity.html) classes.

<br/>

## Features

The code offers two features: (i) integrating a cluster orbits, which is based on the average position and motion of
its cluster members and (ii) integrating orbits for individual stars (independent of cluster membership). 
Each method comes with different statistical sampling methods to estimate uncertainties for the orbit integration.
Regarding the orbital integration, the code allows for customisation of the reference orbit
(default is set to the LSR orbit). Additionally, users are able to choose any potential available at
[galpy.potential](https://docs.galpy.org/en/v1.9.1/potential.html), including custom build potentials.

- For cluster orbit integration:
  - [x] bootstrapping over the cluster members for an average cluster orbit integration
  - [x] methods to facilitate the calculation of averages and uncertainty estimation
  - [ ] _to be implemented_: sampling from a fit to the 6D cluster distribution

- For stellar orbit integration:
  - [x] Monte Carlo-type sampling from a normal distribution based on measurement and measurement uncertainties
  - [x] again, methods to facilitate the calculation of averages and uncertainty estimation

- For visualisation:
  - [ ] _to be implemented_: functions to visualise the resulting data (quick-check)


<br/>

## Installation

This package can be installed locally by cloning the repository and running

```
python3 -m pip install
```

from the top level directory, after which you can import the library to your code.
This is a versioned import package that can be installed locally. This project is partly still under construction and
will be updated accordingly.

<br/><br/>

## How to use StarTracer

More examples can be found on the [ReadTheDocs](https://startracer.readthedocs.io/en/latest/startracer.html)
page for this project. Also, some example code can be found in the **StarTracer directory**.

#### For cluster orbit integration:

1. loading in a data table and initialising a cluster based on this data:

```
>>> from StarTracer.startracer import Cluster, Stars
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

#### For star orbit integrations:

This works similarly for the `Stars` class that allows for orbit sampling from a normal distribution based on
measurement value and uncertainty. However, the resulting array is 4-dimensional and after applying the methods to get
the orbit averages and uncertainties for each star, we are still left with 3 dimensions.
It cannot be stored to a output file yet.

The integrated orbits can be averaged like so:

```
>>> star_orbits = Stars(cluster_data).sample_orbit(10, 0.1, number_of_samples=10000, direction='backward')

>>> mean_orbits = star_orbits.calculate_mean()                       # the mean of all sampled values for each
                                                                     # position/ velocity per timestep
>>> std_orbits = star_orbits.calculate_std()                         # the standard deviation
>>> prctl_orbits = star_orbits.calculate_percentile((14, 86))        # the 14 and 86 percentiles

>>> print(np.shape(mean_orbits), np.shape(std_orbits), np.shape(prctl_orbits))

(50, 7, 101) (50, 7, 101) (2, 50, 7, 101)
```

## License

This package is licensed under the [MIT License](https://choosealicense.com/).
For details, please refer to the License tab at the top of the README page.
