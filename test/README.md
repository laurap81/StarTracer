    .                                        .   --   .                                       .
    .                                    .                .                                   .
    .                                  .                    .                                 .
    *                                 .      StarTracer      .                                *
    .                                 .                      .                                .
    .                                  +                    .                                 .
    .                                                     *                                   .



**StarTracer** is a library that allows to integrate star and cluster orbits with statistical sampling methods in order
to obtain an uncertainty estimation of the traceback result. It is mainly based on
[galpy](https://docs.galpy.org/en/v1.9.1/) which is an [astropy](https://www.astropy.org/index.html)
[affiliated packages](https://www.astropy.org/affiliated/). StarTracer is also based on astropy,
utilising the Table, SkyCoord, and Quantity classes.


###Features


- For cluster orbit integration:
  - bootstrapping over the cluster members for an average cluster orbit integration
  - methods to facilitate the calculation of averages and uncertainty estimation

- For stellar orbit integration:
  - Monte Carlo-type sampling from a normal distribution based on measurement and measurement uncertainties
  - again, methods to facilitate the calculation of averages and uncertainty estimation

- For representation:
  - Functions to visualise the resulting data (quick-check) 


This is a versioned import package that can be installed locally. This project is partly still under construction and
will be updated accordingly.


###Installation

This package can be installed by cloning it and running

```
python3 -m pip install
```

from the top level directory, after which you can import the library to your code.


###Usage Examples

More examples can be found on the ReadTheDocs page of this project.
[!note: insert link]

1. loading in a data table:

```
from astropy.table import Table
cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
cluster = Cluster(cluster_data)
cluster.data.head()
```




