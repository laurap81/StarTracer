import pandas as pd
import numpy as np
import os

# import startracer
from . import startracer


class ClusterGroup:
    """Loops over its input clusters and outputs the distance over time to a reference cluster.

    :param cluster_list: List of clusters with len(list) > 0 for which to calculate the separation to a
        reference cluster from. If clusters provided as numpy.ndarray shape must be
        (>=4 (parameters) x number of timesteps x number of samples). Parameters must include t, X, Y, and Z. They can
        also include U, V, W, which are ignored.
    :type cluster_list: list of SampledCluster instances, list of numpy.ndarray
    :param cluster_label_list: List of cluster labels for distinction in the saved files.
        If list of int is provided, the list will be converted to a list of str.
    :type cluster_label_list: list of int or list of str
    :param cluster_group_name: Name of cluster group for saved-file identification.
    :type: cluster_group_name:str

    :raises ValueError: If cluster_label_list has a different length than cluster_list.
    """
    def __init__(self, cluster_list, cluster_label_list, cluster_group_name):
        """Constructor method
        """
        self.separation = None
        self.average_dataframe = None
        self.group_name = cluster_group_name

        if isinstance(cluster_list[0], np.ndarray):
            if np.shape(cluster_list[0])[0] >= 4:
                pass
            else:
                raise ValueError("Cluster array must have shape (>=4 x number of timesteps x number of samples).")

            self.cluster_list = cluster_list

        elif isinstance(cluster_list[0], startracer.SampledCluster):
            cluster_to_array_list = [clr.get_data() for clr in cluster_list]
            self.cluster_list = cluster_to_array_list

        else:
            raise TypeError("Clusters in cluster_list need to be either numpy.ndarray or SampledCluster instance.")

        if len(self.cluster_list) != len(cluster_label_list):
            raise ValueError(f"Cluster list ({len(cluster_list)}) needs to be same length as"
                             f"cluster label list ({len(cluster_label_list)}).")
        else:
            if all(isinstance(cl, int) for cl in cluster_label_list):
                self.cluster_label_list = [str(lbl) for lbl in cluster_label_list]
            else:
                self.cluster_label_list = cluster_label_list

    def calculate_cluster_separation(self, reference_cluster,
                                     average_method='median', uncertainty_method='mad',
                                     save_collected_average=True, save_single_average=False, save_single_sampled=False,
                                     return_collected_array=False, print_out=False):
        """Calculates the distance between the clusters in ClusterGroup and a reference cluster.

        Offers to save the average separation and uncertainty or the not averaged sampled values for each cluster.
        It also offers to save the average separation for all clusters in one 'csv' file, distinguished by the
        cluster_label input parameter.
        The average and uncertainty can be either median or mean and median average deviation or
        standard deviation, respectively.

        :param reference_cluster: Reference cluster data as numpy.ndarray or SampledCluster instance.
        :type reference_cluster: numpy.ndarray, SampledCluster
        :param average_method: 'median' (default) or 'mean'
        :type average_method: str, optional
        :param uncertainty_method: Median absolut deviation: 'mad' (default) or standard deviation 'std'.
        :type uncertainty_method: str, optional
        :param save_collected_average: If True (default), saves the separation over time between each cluster and the
            reference cluster in one 'csv' file.
        :type save_collected_average: bool, optional
        :param save_single_average: If True, a csv file for each cluster's separation average and uncertainty is saved
            with the cluster label in the file name. Defaults to False.
        :type save_single_average: bool, optional
        :param save_single_sampled: If True, a csv file for all sampled separation values (not averaged) of each cluster
            is saved with the cluster label in the file name. Defaults to False.
        :type save_single_sampled: bool, optional
        :param return_collected_array: If True, a 3-dimensional array with all stored separation values for each
            timestep and cluster is returned. Shape is (number of clusters x number of timesteps x number of samples).
            Defaults to False.
        :type return_collected_array: bool, optional
        :param print_out: If True, prints progress statements in console. Defaults to False.
        :type print_out: bool, optional

        :return: If ``return_collected_array`` is set to True, returns separation array for all clusters. Also stored in
            :attr:`ClusterGroup.separation` after running the method. Shape is (number of clusters x number of timesteps
            x number of samples)
        :rtype: numpy.ndarray, optional

        :raises ValueError: If reference cluster does not have the shape (7 x any x any).
        :raises TypeError: If reference cluster is neither numpy.ndarray nor SampledCluster instance.
        :raises ValueError: If average_method and uncertainty_method is neither 'median' nor 'mean'
            and 'mad' nor 'std', respectively.

        |

        Examples
        =========

        .. code-block::

        >>> from StarTracer.startracer import Cluster
        >>> path_to_table = './example_data/ExampleCluster_1.csv'
        >>> path_to_table_2 = './example_data/ExampleCluster_2.csv'
        >>> # initialising both clusters
        >>> cluster_1 = Cluster(path_to_table)
        >>> cluster_2 = Cluster(path_to_table_2)

        >>> # sampling orbit traceback for both clusters
        >>> sampled_cluster_1 = cluster_1.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')
        >>> sampled_cluster_2 = cluster_2.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')

        >>> # calculating cluster separation to reference cluster (cluster_2)
        >>> # for each cluster (here only cluster_1), timestep and sample
        >>> group_1 = ClusterGroup(cluster_list=[sampled_cluster_1], cluster_label_list=['01'],
        ...                         cluster_group_name='group1')
        >>> separation_cluster_1_2 = group_1.calculate_cluster_separation(reference_cluster=sampled_cluster_2,
        ...                                                                return_collected_array=True)
        >>> np.shape(separation_cluster_1_2)
        (1, 101, 1000)
        >>> group_1.average_dataframe.head()[:3]
              t  01_median    01_mad
        0 -10.0  17.294575  5.025454
        1  -9.9  17.050660  4.970929
        2  -9.8  16.812033  4.923569
        """
        separation_shape = (len(self.cluster_list),) + np.shape(self.cluster_list[0])[1:]
        self.separation = np.zeros(separation_shape)

        if isinstance(reference_cluster, np.ndarray):
            if np.shape(reference_cluster)[0] >= 4:
                pass
            else:
                raise ValueError("Reference cluster array must have shape"
                                 "(>=4 x number of timesteps x number of samples).")

        elif isinstance(reference_cluster, startracer.SampledCluster):
            reference_cluster = reference_cluster.get_data()

        else:
            raise TypeError("Reference cluster needs to be either numpy.ndarray or SampledCluster instance.")

        average_separation_dataframe = pd.DataFrame(reference_cluster[0, :, 0], columns=['t'])
        # column_list = []

        for i_clr, cluster in enumerate(self.cluster_list):
            cluster_label = self.cluster_label_list[i_clr]

            x_distance = np.subtract(reference_cluster[1, :, :], cluster[1, :, :])
            y_distance = np.subtract(reference_cluster[2, :, :], cluster[2, :, :])
            z_distance = np.subtract(reference_cluster[3, :, :], cluster[3, :, :])

            separation_array = np.sqrt(np.power(x_distance, 2) + np.power(y_distance, 2) + np.power(z_distance, 2))
            self.separation[i_clr, :, :] = separation_array

            if average_method == 'median':
                separation_average = np.nanmedian(separation_array, axis=1)
                average_separation_dataframe[f'{cluster_label}_{average_method}'] = separation_average
            elif average_method == 'mean':
                separation_average = np.nanmean(separation_array, axis=1)
                average_separation_dataframe[f'{cluster_label}_{average_method}'] = separation_average
            else:
                raise ValueError(f"average_method can only be 'median' (default) or 'mean',"
                                 f"not {average_method}.")
            if uncertainty_method == 'mad':
                separation_uncertainty = np.nanmedian(np.abs(np.subtract(
                    separation_array, np.nanmean(separation_array, axis=1)[:, None])), axis=1)
                average_separation_dataframe[f'{cluster_label}_{uncertainty_method}'] = separation_uncertainty
            elif uncertainty_method == 'std':
                separation_uncertainty = np.nanstd(separation_array, axis=1)
                average_separation_dataframe[f'{cluster_label}_{uncertainty_method}'] = separation_uncertainty
            else:
                raise ValueError(f"uncertainty_method can only be 'mad' (default) or 'std',"
                                 f"not {uncertainty_method}.")

            if save_single_average:
                average_column = f'{cluster_label}_{average_method}'
                uncertainty_column = f'{cluster_label}_{uncertainty_method}'
                df_single_save = average_separation_dataframe[['t', average_column, uncertainty_column]]
                try:
                    os.makedirs('../Tables/ClusterSeparation/')
                except FileExistsError:
                    # directory already exists
                    pass
                if print_out:
                    print(f'...saving table at ../Tables/ClusterSeparation/{cluster_label}_AverageSeparation.csv')
                    print('-' * 100)
                df_single_save.to_csv(f'../Tables/ClusterSeparation/{cluster_label}_AverageSeparation.csv', index=False)

            if save_single_sampled:
                try:
                    os.makedirs('../Tables/ClusterSeparation/')
                except FileExistsError:
                    # directory already exists
                    pass
                if print_out:
                    print(f'...saving table at ../Tables/ClusterSeparation/{cluster_label}_SampledSeparation.csv')
                    print('-' * 100)
                np.savetxt(f'../Tables/ClusterSeparation/{cluster_label}_SampledSeparation.csv',
                           separation_array, delimiter=",")

        self.average_dataframe = average_separation_dataframe

        if save_collected_average:
            try:
                os.makedirs('../Tables/ClusterSeparation/')
            except FileExistsError:
                # directory already exists
                pass
            if print_out:
                print(f'...saving table at ../Tables/'
                      f'ClusterSeparation/{self.group_name}_Collected_AverageSeparation.csv')
                print('-' * 100)
            average_separation_dataframe.to_csv(
                f'../Tables/ClusterSeparation/{self.group_name}_Collected_AverageSeparation.csv', index=False)

        if return_collected_array:
            return self.separation
