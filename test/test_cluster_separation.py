import sys
from io import StringIO
import pytest
import numpy as np
from astropy.table import Table

import StarTracer.startracer as st
import StarTracer.separation as sp

test_data = Table.read('./StarTracer/example_data/ExampleCluster_1.fits')
test_cluster = st.Cluster(test_data)
test_sampled_cluster = test_cluster.sample_orbit(5, 0.1)

test_sampled_reference_cluster = st.Cluster('./StarTracer/example_data/ExampleCluster_2.fits').sample_orbit(5, 0.1)


def test_initialisation():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    assert isinstance(test_group.cluster_list[0], np.ndarray) & isinstance(test_group.cluster_label_list[0], str)


def test_cluster_separation():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    separation_1_2 = test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=True,
                                                             return_collected_array=True)
    shape_0 = len(test_group.cluster_list)
    shape_1 = (int(5 / 0.1) * 2) + 1
    shape_2 = 1000
    assert np.shape(separation_1_2) == (shape_0, shape_1, 1000)


def test_cluster_separation_dataframe():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=False,
                                            return_collected_array=False)
    column_names = test_group.average_dataframe.columns
    assert ('1_median' in column_names) & ('1_mad' in column_names)


def test_cluster_separation_dataframe_average():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=False, average_method='mean',
                                            uncertainty_method='std')
    column_names = test_group.average_dataframe.columns
    assert ('1_mean' in column_names) & ('1_std' in column_names)


def test_cluster_separation_single_average():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    captured_output = StringIO()
    sys.stdout = captured_output
    test_group.calculate_cluster_separation(test_sampled_cluster, save_single_average=True,
                                            save_collected_average=False, print_out=True)

    sys.stdout = sys.__stdout__
    cogv = captured_output.getvalue()

    assert '1_AverageSeparation' in cogv


def test_cluster_separation_single_sampled():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    captured_output = StringIO()
    sys.stdout = captured_output
    test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=False,
                                            save_single_sampled=True, print_out=True)

    sys.stdout = sys.__stdout__
    cogv = captured_output.getvalue()

    assert '1_SampledSeparation' in cogv


def test_cluster_separation_collected_average():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    captured_output = StringIO()
    sys.stdout = captured_output
    test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=True,
                                            print_out=True)

    sys.stdout = sys.__stdout__
    cogv = captured_output.getvalue()

    assert 'Grp1_Collected_AverageSeparation' in cogv


def test_separation_error_average():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    with pytest.raises(ValueError, match=r"average_method.*"):
        test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=False,
                                                average_method='none')


def test_separation_error_uncertainty():
    test_group = sp.ClusterGroup(cluster_list=[test_sampled_cluster], cluster_label_list=[1], cluster_group_name='Grp1')

    with pytest.raises(ValueError, match=r"uncertainty_method.*"):
        test_group.calculate_cluster_separation(test_sampled_cluster, save_collected_average=False,
                                                uncertainty_method='none')
