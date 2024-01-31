from startracer import Cluster, Stars
import numpy as np
import matplotlib.pyplot as plt

# using some example data, randomly sampled from a normal distribution
path_to_table = './example_data/ExampleCluster_1.csv'

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
# Integrating a cluster with bootstrapping and plotting the median of all
# sampled orbits on top of a fraction of all integrated ones
cluster2 = Stars(path_to_table)

# cluster2

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


star_orbits = Stars(path_to_table).sample_orbit(10, 0.1, number_of_samples=10000, direction='backward')

mean_orbits = star_orbits.calculate_mean()                       # the mean of all sampled values for each
                                                                 # position/ velocity per timestep
std_orbits = star_orbits.calculate_std()                         # the standard deviation
prctl_orbits = star_orbits.calculate_percentile((14, 86))        # the 14 and 86 percentiles

print(np.shape(mean_orbits), np.shape(std_orbits), np.shape(prctl_orbits))
