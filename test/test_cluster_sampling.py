from astropy.table import Table, QTable
import pandas as pd
import numpy as np
from StarTracer.startracer import Cluster, SampledCluster

test_data = Table.read('./StarTracer/example_data/ExampleCluster_1.fits')
test_cluster = Cluster(test_data)


def test_cluster_type():
    assert isinstance(test_cluster.data, pd.DataFrame)


def test_sampled_cluster_shape_both():
    time_end = 5
    time_step = 0.1
    number_of_samples = 100
    sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step,
                                                       number_of_samples=number_of_samples, direction='both')

    sampled_cluster_shape = np.shape(sampled_cluster_orbits.get_data())
    dim_0 = 7
    dim_1 = int(time_end / time_step) * 2 + 1
    dim_2 = number_of_samples

    assert sampled_cluster_shape == (dim_0, dim_1, dim_2)


def test_sampled_cluster_shape_backward():
    time_end = 5
    time_step = 0.1
    number_of_samples = 100
    sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step,
                                                       number_of_samples=number_of_samples, direction='backward',
                                                       print_out=True)

    sampled_cluster_shape = np.shape(sampled_cluster_orbits.get_data())
    dim_0 = 7
    dim_1 = int(time_end / time_step) + 1
    dim_2 = number_of_samples

    assert sampled_cluster_shape == (dim_0, dim_1, dim_2)


def test_reference_orbit():
    time_end = 5
    time_step = 0.1
    number_of_samples = 100
    sampled_stars_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step,
                                                     number_of_samples=number_of_samples, direction='forward',
                                                     reference_orbit_lsr=False,
                                                     reference_object_pv=[0, 0, 0, 1, 1, 1])

    sampled_stars_shape = np.shape(sampled_stars_orbits.get_data())
    dim_0 = 7
    dim_1 = int(time_end / time_step) + 1
    dim_2 = number_of_samples

    assert sampled_stars_shape == (dim_0, dim_1, dim_2)


def test_average_method():
    time_end = 5
    time_step = 0.1
    number_of_samples = 100
    sampled_cluster_orbits = test_cluster.sample_orbit(time_end=time_end, time_step=time_step,
                                                       number_of_samples=number_of_samples, direction='forward',
                                                       average_method='mean')

    sampled_cluster_shape = np.shape(sampled_cluster_orbits.get_data())
    dim_0 = 7
    dim_1 = int(time_end / time_step) + 1
    dim_2 = number_of_samples

    assert sampled_cluster_shape == (dim_0, dim_1, dim_2)


def test_get_sampled_data():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)

    assert isinstance(sampled_cluster.get_data(), np.ndarray)


def test_add_multiple():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    sampled_cluster.add_mean()
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_std()
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_mean' in sampled_cluster.summary_dataframe.columns) & \
           ('X_std' in sampled_cluster.summary_dataframe.columns)


def test_mean_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_mean()
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_mean' in sampled_cluster.summary_dataframe.columns)


def test_median_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_median()
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_median' in sampled_cluster.summary_dataframe.columns)


def test_mad_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_mad()
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_mad' in sampled_cluster.summary_dataframe.columns)


def test_std_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_std()
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_std' in sampled_cluster.summary_dataframe.columns)


def test_percentile_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_percentile((50))
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 6) & \
           ('X_p50' in sampled_cluster.summary_dataframe.columns)


def test_two_percentile_added():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    summary_df_before = len(sampled_cluster.summary_dataframe.columns)

    sampled_cluster.add_percentile((25, 75))
    summary_df_after = len(sampled_cluster.summary_dataframe.columns)

    assert ((summary_df_after - summary_df_before) == 12) & \
           ('X_p25' in sampled_cluster.summary_dataframe.columns) & \
           ('X_p75' in sampled_cluster.summary_dataframe.columns)


def test_qtable():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    sampled_cluster.add_mean()

    qtable = sampled_cluster.to_astropy_table(include_units=True)
    qtable_columns = len(qtable.columns)
    assert (isinstance(qtable, QTable)) & (qtable_columns > 1)


def test_table():
    time_end = 5
    time_step = 0.1
    sampled_cluster = test_cluster.sample_orbit(time_end, time_step)
    sampled_cluster.add_mean()

    table = sampled_cluster.to_astropy_table(include_units=False)
    table_columns = len(table.columns)
    assert (isinstance(table, Table)) & (table_columns > 1)
