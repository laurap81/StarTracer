from astropy.table import Table
import pandas as pd
import numpy as np
import pytest
import sys

sys.path.append('/Users/laura/Lectures/23WS/OpenSource/StarTracer')

from StarTracer.startracer import Cluster, SampledCluster

# data_path = '/Users/laura/Lectures/23WS/OpenSource/StarTracer/ExampleData/ExampleCluster.fits'
# table = Table.read(data_path)

test_data = Table.read('/Users/laura/Lectures/23WS/OpenSource/StarTracer/ExampleData/ExampleCluster.fits')
test_cluster = Cluster(test_data)


def test_check_direction():
    time_end = 5
    time_step = 0.1

    with pytest.raises(ValueError, match=r"direction.*"):
        sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step, direction='none')


def test_check_averagemethod():
    time_end = 5
    time_step = 0.1

    with pytest.raises(ValueError, match=r"average_method.*"):
        sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step,
                                                           average_method='none')


def test_check_timeend():
    time_end = 0.0
    time_step = 0.0

    with pytest.raises(ValueError, match=r"time_end.*"):
        sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step)


def test_check_timestep():
    time_end = - 1.0
    time_step = 0.0

    with pytest.raises(ValueError, match=r"time_step.*"):
        sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step)


def test_check_timestep_smaller_timeend():
    time_end = 1.0
    time_step = -2.0

    with pytest.raises(ValueError, match=r"time_step .* time_end"):
        sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step)


def test_save_filetype():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)

    with pytest.raises(ValueError, match=r"file_type.*"):
        sampled_cluster.save_table(path_to_file='./01.csv', file_type='none')
