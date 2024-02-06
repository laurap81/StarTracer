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


**StarTracer** is a Python package designed for integrating star and cluster orbits using statistical sampling methods.
Its primary purpose is to provide an uncertainty estimation of the traceback result. It is mainly based on
[galpy](https://docs.galpy.org/en/v1.9.1/) which is an [astropy](https://www.astropy.org/index.html)
[affiliated package](https://www.astropy.org/affiliated/). StarTracer is also based on astropy,
utilising the [Table](https://docs.astropy.org/en/stable/api/astropy.table.Table.html),
[SkyCoord](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html),
and [Quantity](https://docs.astropy.org/en/stable/units/quantity.html) classes.

<br/>

## Features

The code provides two main features: (i) Integration of cluster orbits, which is based on the average position and
motion of its cluster members. (ii) Integration of orbits for individual stars, independent of cluster membership.

Each feature uses an appropriate statistical sampling method to estimate uncertainties for the orbit integration.
In terms of orbital integration, the code allows users to customize the reference orbit, with the default set to the
LSR orbit. Additionally, users have the flexibility to choose any potential available at
[galpy.potential](https://docs.galpy.org/en/v1.9.1/potential.html), including custom-built potentials.

[!NOTE]
EDIT 02.2024: A supplementary module named "separation" has been incorporated into the code. This module introduces
a method for calculating the separation between a reference cluster and one or more clusters in a group.

- For cluster orbit integration:
  - [x] bootstrapping over the cluster members for an average cluster orbit integration
  - [x] methods to facilitate the calculation of averages and uncertainty estimation
  - [ ] _to be implemented_: sampling from a fit to the 6D cluster distribution

- For stellar orbit integration:
  - [x] Monte Carlo-type sampling from a normal distribution based on measurement and measurement uncertainties
  - [x] again, methods to facilitate the calculation of averages and uncertainty estimation

- For orbit analysis:
  - [x] A method to calculate the spatial separation between several clusters and one reference frame over time
  - [ ] _to be implemented_: the same method to calculate the spatial separation between all stars over time

- For visualisation:
  - [ ] _to be implemented_: functions to visualise the resulting data (quick-check)


<br/>

## Installation

This package can be installed locally by cloning the repository and running

```
python3 -m pip install .
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

>>> sampled_cluster.summary_dataframe.columns

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

>>> np.shape(mean_orbits), np.shape(std_orbits), np.shape(prctl_orbits)

((50, 7, 101) (50, 7, 101) (2, 50, 7, 101))
```

#### Positional cluster separation over time:

Using the ``separation.py`` module, import the ClusterGroup class. The ClusterGroup can be initialised with one or more
clusters. Its ``calculate_cluster_separation()`` method calculates the spatial separation over time to a
reference cluster, that is an input of the method. Separation can be saved for each cluster (averaged or not averaged)
or averaged separation for each cluster in one 'csv' file together.

```
>>> from StarTracer.startracer import Cluster
>>> from StarTracer.separation import ClusterGroup
>>> import numpy as np

>>> # initialising both clusters
>>> cluster_1_data = Table.read('./example_data/ExampleCluster_1.fits')
>>> cluster_1 = Cluster(cluster_1_data)
>>> cluster_2_data = Table.read('./example_data/ExampleCluster_2.fits')
>>> cluster_2 = Cluster(cluster_2_data)

>>> # sampling orbit traceback for both clusters
>>> sampled_cluster_1 = cluster_1.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')
>>> sampled_cluster_2 = cluster_2.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')

>>> # calculating cluster sepration to reference cluster (cluster_2)
>>> # for each cluster (here only cluster_1), timestep and sample
>>> group_1 = ClusterGroup(cluster_list=[sampled_cluster_1], cluster_label_list=['01'], cluster_group_name='group1')
>>> separation_cluster_1_2 = group_1.calculate_cluster_separation(reference_cluster=sampled_cluster_2,
                                                                  return_collected_array=True)
>>> np.shape(separation_cluster_1_2)

(1, 101, 1000)

>>> group_1.average_dataframe.head()

        t  01_median    01_mad
0   -10.0  17.786195  4.630371
1    -9.9  17.549027  4.584467
2    -9.8  17.321228  4.534348
3    -9.7  17.095397  4.485154
4    -9.6  16.865780  4.441097
..    ...        ...       ...
96   -0.4   8.205172  1.119309
97   -0.3   8.438620  1.097481
98   -0.2   8.664430  1.082462
99   -0.1   8.884338  1.081523
100  -0.0   9.095192  1.076424

[101 rows x 3 columns]>                                                        
```


## License

This package is licensed under the [MIT License](https://choosealicense.com/).
For details, please refer to the License tab at the top of the README page.
