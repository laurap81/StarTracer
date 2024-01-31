from astropy.table import Table, QTable
import pandas as pd
import numpy as np
from StarTracer.startracer import Stars, SampledStars

filepath = './StarTracer/example_data/ExampleCluster_1.fits'
test_data = Table.read(filepath)
test_stars = Stars(test_data)


def test_stars_type():
    assert isinstance(test_stars.data, pd.DataFrame)


def test_stars_type_fromstring():
    new_stars = Stars(filepath)
    assert isinstance(new_stars.data, pd.DataFrame)

#
# def test_sampled_stars_shape_both():
#     time_end = 5
#     time_step = 0.1
#     number_of_samples = 100
#     sampled_stars_orbits = test_stars.sample_orbit(time_end=time_end, time_step=time_step,
#                                                    number_of_samples=number_of_samples, direction='both')
#
#     sampled_stars_shape = np.shape(sampled_stars_orbits.get_data())
#     dim_0 = len(test_stars.data)
#     dim_1 = 7
#     dim_2 = int(time_end / time_step) * 2 + 1
#     dim_3 = number_of_samples
#
#     assert sampled_stars_shape == (dim_0, dim_1, dim_2, dim_3)
#
#
# def test_sampled_stars_shape_backward():
#     time_end = 5
#     time_step = 0.1
#     number_of_samples = 100
#     sampled_stars_orbits = test_stars.sample_orbit(time_end=time_end, time_step=time_step,
#                                                    number_of_samples=number_of_samples, direction='backward')
#
#     sampled_stars_shape = np.shape(sampled_stars_orbits.get_data())
#     dim_0 = len(test_stars.data)
#     dim_1 = 7
#     dim_2 = int(time_end / time_step) + 1
#     dim_3 = number_of_samples
#
#     assert sampled_stars_shape == (dim_0, dim_1, dim_2, dim_3)
#
#
# def test_get_sampled_data():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#
#     assert isinstance(sampled_stars.get_data(), np.ndarray)
#
#
# def test_mean_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#     mean_orbit = sampled_stars.mean_array()
#
#     assert np.shape(mean_orbit) == np.shape(sampled_orbit_data)[:-1]
#
#
# def test_median_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#     median_orbit = sampled_stars.median_array()
#
#     assert np.shape(median_orbit) == np.shape(sampled_orbit_data)[:-1]
#
#
# def test_mad_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#     mad_orbit = sampled_stars.mad_array()
#
#     assert np.shape(mad_orbit) == np.shape(sampled_orbit_data)[:-1]
#
#
# def test_std_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#     std_orbit = sampled_stars.std_array()
#
#     assert np.shape(std_orbit) == np.shape(sampled_orbit_data)[:-1]
#
#
# def test_percentile_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#     percentile_orbit = sampled_stars.percentile_array(50)
#
#     assert np.shape(percentile_orbit) == np.shape(sampled_orbit_data)[:-1]
#
#
# def test_two_percentile_array():
#     time_end = 5
#     time_step = 0.1
#     sampled_stars = test_stars.sample_orbit(time_end, time_step)
#     sampled_orbit_data = sampled_stars.get_data()
#
#     percentile_orbit = sampled_stars.percentile_array((25, 75))
#
#     assert np.shape(percentile_orbit) == (2,) + np.shape(sampled_orbit_data)
#
#
# def test_reference_orbit():
#     time_end = 5
#     time_step = 0.1
#     number_of_samples = 100
#     sampled_stars_orbits = test_stars.sample_orbit(time_end=time_end, time_step=time_step,
#                                                    number_of_samples=number_of_samples, direction='both',
#                                                    reference_orbit_lsr=False,
#                                                    reference_object_pv=[0, 0, 0, 1, 1, 1])
#
#     sampled_stars_shape = np.shape(sampled_stars_orbits.get_data())
#     dim_0 = len(test_stars.data)
#     dim_1 = 7
#     dim_2 = int(time_end / time_step) * 2 + 1
#     dim_3 = number_of_samples
#
#     assert sampled_stars_shape == (dim_0, dim_1, dim_2, dim_3)
