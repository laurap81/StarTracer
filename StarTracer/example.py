from startracer import Cluster, Stars
from separation import ClusterGroup

import numpy as np
import matplotlib.pyplot as plt

# using some example data, randomly sampled from a normal distribution
path_to_table = './example_data/ExampleCluster_1.csv'
path_to_table_2 = './example_data/ExampleCluster_2.csv'

##################################################################################
# EXAMPLE 1                                                                      #
##################################################################################
# Integrating a cluster with bootstrapping and plotting the median of all
# sampled orbits on top of a fraction of all integrated ones
cluster1 = Cluster(path_to_table)

time_end = 5
time_step = 0.1
number_of_samples = 100

cluster1_orbits = cluster1.sample_orbit(time_end, time_step, number_of_samples,
                                        direction='backward', average_method='median',
                                        # reference_orbit_lsr=True, reference_object_pv=None,
                                        # potential=MWPotential2014,
                                        print_out=True)
print(np.shape(cluster1_orbits.get_data()))

# calculating median and storing it in the summary_dataframe
cluster1_orbits.add_median()
median_orbits = cluster1_orbits.summary_dataframe
all_orbits = cluster1_orbits.get_data()

f1 = plt.Figure()
ax = f1.add_subplot(111)
ax.set_xlabel('lookback time  --  Myr')
ax.set_ylabel('Z axis  --  pc')
for nos in range(int(number_of_samples / 2)):
    ax.plot(all_orbits[0, :, nos], all_orbits[3, :, nos], 'tab:gray', alpha=0.3, lw=0.5)
ax.plot(median_orbits['t'], median_orbits['Z_median'], 'tab:red')

f1.tight_layout()
# f1.savefig('./example_data/Cluster1_Traceback.png', format='png', dpi=250)
# plt.show()

##################################################################################
# EXAMPLE 2                                                                      #
##################################################################################
# Integrating all stars of a cluster with MC-type sampling from the stars
# measurements and measurement uncertainties and plotting each star's orbit over
# a random fraction of each star's integrated orbits
cluster2 = Stars(path_to_table)

time_end = 5
time_step = 0.1
number_of_samples = 100

cluster2_orbits = cluster2.sample_orbit(time_end, time_step, number_of_samples,
                                        direction='both',
                                        # reference_orbit_lsr=True, reference_object_pv=None,
                                        # potential=MWPotential2014,
                                        print_out=True)
print(np.shape(cluster2_orbits.get_data()))

mean_orbits = cluster2_orbits.calculate_mean()
all_stars_orbits = cluster2_orbits.get_data()

f2 = plt.Figure()
ax = f2.add_subplot(111)
ax.set_xlabel('lookback time  --  Myr')
ax.set_ylabel('Z axis  --  pc')
for strs in range(len(cluster2.data.index)):
    for nos in range(int(number_of_samples / 2)):
        ax.plot(all_stars_orbits[strs, 0, :, nos], all_stars_orbits[strs, 3, :, nos], 'tab:gray', alpha=0.03, lw=0.5)
    ax.plot(mean_orbits[strs, 0, :], mean_orbits[strs, 3, :], 'tab:red')

f2.tight_layout()
# f2.savefig('./example_data/Cluster2_StarTraceback.png', format='png', dpi=250)
# plt.show()


##################################################################################
# EXAMPLE 3                                                                      #
##################################################################################
# calculating the stats of the sampled orbits of several stars in
# the example cluster 1
# star_orbits = Stars(path_to_table).sample_orbit(10, 0.1, number_of_samples=10000, direction='backward')
#
# mean_orbits = star_orbits.calculate_mean()  # the mean of all sampled values for each
# # position/ velocity per timestep
# std_orbits = star_orbits.calculate_std()  # the standard deviation
# prctl_orbits = star_orbits.calculate_percentile((14, 86))  # the 14 and 86 percentiles
#
# print(np.shape(mean_orbits), np.shape(std_orbits), np.shape(prctl_orbits))


##################################################################################
# EXAMPLE 4                                                                      #
##################################################################################
# calculating the separation between cluster_1 and cluster_2 over time
# initialising both clusters
cluster_1 = Cluster(path_to_table)
cluster_2 = Cluster(path_to_table_2)

# sampling orbit traceback for both clusters
sampled_cluster_1 = cluster_1.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')
sampled_cluster_2 = cluster_2.sample_orbit(10, 0.1, number_of_samples=1000, direction='backward')

# calculating cluster separation to reference cluster (cluster_2)
# for each cluster (here only cluster_1), timestep and sample
group_1 = ClusterGroup(cluster_list=[sampled_cluster_1], cluster_label_list=['01'], cluster_group_name='group1')
separation_cluster_1_2 = group_1.calculate_cluster_separation(reference_cluster=sampled_cluster_2,
                                                              return_collected_array=True)
print(np.shape(separation_cluster_1_2))
print(group_1.average_dataframe.head()[:3])

