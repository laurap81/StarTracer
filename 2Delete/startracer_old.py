from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from astropy.coordinates import Galactocentric, LSR, Distance, SkyCoord
import astropy.units as unit
from astropy.table import Table

import pandas as pd
import warnings
import numpy as np


# todo: just realized I can't let them pass in a skycoord because I need to use errors
#  -> let them pass in path or table like initially planned with data and errors.
#  Sample from the data and errors and pass them to the traceback functions as lists of Quantities

# todo: also, simply return table of 'cluster: timestep x sample' | 'single: timestep x star x sample'
#  and add another funtion to analyze (mean, median, std, mad, percentile)

# todo: maybe add plot function

# class StarTracer:
#     def __init__(self, realpart, imagpart):
#         self.r = realpart
#         self.i = imagpart


########################################################################################################################
def mad(array, axis=1):
    """ median absolut deviation
    function calculates the median absolut deviation of a 1 to 3D array along either axis.
    :param array: Array of which to calculate the median absolut deviation of.
    :type array: array-like
    :param axis: 0 or 1 for a 2D array. 0 for a 1D array. Calculates mean and median along this axis.
    default is axis=1 (rows)
    :type axis: int
    :return: median absolut deviation of array
    :rtype: array, float
    """
    if axis == 0:
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(array, np.nanmean(array, axis=0))), axis=0)
    elif axis == 1:
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(array, np.nanmean(array, axis=1)[:, None])), axis=1)
    elif axis == 2:
        median_absolut_deviation = np.nanmedian(np.abs(
            np.subtract(array, np.nanmean(array, axis=2)[:, :, None])), axis=2)
    elif axis == 3:
        median_absolut_deviation = np.nanmedian(np.abs(
            np.subtract(array, np.nanmean(array, axis=3)[:, :, :, None])), axis=3)
    else:
        raise 'dimension not implemented'
    return median_absolut_deviation


########################################################################################################################


########################################################################################################################
def skycoord_from_table(path_to_file):
    """
    create a 6D astropy.coordinates.SkyCoord from the input table
    (using ra, dec, parallax/ distance, pmra, pmdec, radial velocity).
    If no column is called 'distance', parallax is automatically converted to distance
    :param path_to_file: path to table file
    :type path_to_file: str
    :return: 6D SkyCoord object based on the data in the table
    :rtype: astropy.coordinates.SkyCoord
    """
    itable = Table.read(path_to_file)
    column_names = itable.colnames

    if 'distance' in column_names:
        dist = itable['distance'].value * unit.kpc
    elif ('distance' not in column_names) & ('parallax' in column_names):
        dist = Distance(itable['parallax']).to_value(unit.kpc)
    else:
        raise 'Table has no column named "distance" or "parallax".'

    skycoord_object = SkyCoord(ra=unit.Quantity(itable['ra'].value * unit.deg, copy=False),
                               dec=unit.Quantity(itable['dec'].value * unit.deg, copy=False),
                               distance=unit.Quantity(dist, unit.kpc, copy=False),
                               pmra=unit.Quantity(itable['pmra'].value * unit.mas / unit.yr, copy=False),
                               pmdec=unit.Quantity(itable['pmdec'].value * unit.mas / unit.yr, copy=False),
                               radial_velocity=unit.Quantity(itable['radial_velocity'].value * unit.km / unit.s))

    return skycoord_object


########################################################################################################################


########################################################################################################################
def read_table_to_df(filepath_or_table):
    """
    reading in a table and/or converting the input to a pandas dataframe, if it is not already.
    :param filepath_or_table: path to saved file, or table/ array. All versions need headers including
    ['ra', 'ra_error', 'dec', 'dec_error', 'distance', 'distance_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
    'radial_velocity', 'radial_velocity_error']. If input is an array, this order must be kept.
    :type filepath_or_table: str, np.ndarray, pd.DataFrame, astropy.table.Table
    :return: dataframe of input
    :rtype: pd.DataFrame
    """
    column_names = ['ra', 'ra_error', 'dec', 'dec_error', 'distance',
                    'distance_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                    'radial_velocity', 'radial_velocity_error']

    # if input is a str it is assumed to be a path to the file location
    # file is loaded as Table, making it independent of file format
    if isinstance(filepath_or_table, str):
        itable = Table.read(filepath_or_table)
        df = itable.to_pandas()

    # if input is a astropy.table.Table, it is converted to dataframe
    elif isinstance(filepath_or_table, Table):
        df = filepath_or_table.to_pandas()

    # if input is a dataframe, it will be returned as such
    elif isinstance(filepath_or_table, pd.DataFrame):
        df = filepath_or_table

    # if input is a numpy array it is assumed to be the input data and converted to a dataframe,
    # one dimension needs to match the number of columns (6D coordinates + errors -> 12 columns)
    elif isinstance(filepath_or_table, np.ndarray) & ((np.shape(filepath_or_table)[0] == len(column_names))
                                                      | (np.shape(filepath_or_table)[1] == len(column_names))):
        if np.shape(filepath_or_table)[0] == len(column_names):
            df = pd.DataFrame(np.transpose(filepath_or_table), columns=column_names)
        else:
            df = pd.DataFrame(filepath_or_table, columns=column_names)

    else:
        raise 'ExampleData input must be either a path to a table, a pandas DataFrame,' \
              'or a 2D array with shape being either (Nx12) or (12xN).'

    return df


########################################################################################################################


########################################################################################################################
def cluster_average(cluster_orbit_array, average_method='median', uncertainty_evaluation='mad', save=False,
                    save_path='Average_Cluster_Orbit.csv'):
    """ Analysing the cluster traceback

    :param cluster_orbit_array:
    :type cluster_orbit_array: array-like
    :param average_method:
    :type average_method: str
    :param uncertainty_evaluation:
    :type uncertainty_evaluation: str
    :param save:
    :type save: bool
    :param save_path:
    :type save_path: str
    :return:
    :rtype:
    """

    df_average = pd.DataFrame()
    if average_method == 'median':
        df_average[['t', 'X_median', 'Y_median', 'Z_median', 'U_median', 'V_median', 'W_median']] = \
            np.transpose(np.nanmedian(cluster_orbit_array, axis=2))
    elif average_method == 'mean':
        df_average[['t', 'X_mean', 'Y_mean', 'Z_mean', 'U_mean', 'V_mean', 'W_mean']] = \
            np.transpose(np.nanmean(cluster_orbit_array, axis=2))
    else:
        raise 'average_method can either "median" or "mean".'

    if uncertainty_evaluation == 'mad':
        df_average[['X_mad', 'Y_mad', 'Z_mad', 'U_mad', 'V_mad', 'W_mad']] = \
            np.transpose(mad(cluster_orbit_array[1:, :, :], axis=2))
    elif uncertainty_evaluation == 'std':
        df_average[['X_std', 'Y_std', 'Z_std', 'U_std', 'V_std', 'W_std']] = \
            np.transpose(np.nanstd(cluster_orbit_array[1:, :, :], axis=2))
    else:
        raise 'uncertainty_evaluation can either "mad" or "std".'

    if save:
        df_average.to_csv(save_path)

    return df_average


########################################################################################################################


########################################################################################################################
def star_average(star_orbit_array, average_method='mean', uncertainty_evaluation='std', save=False,
                 save_path='Average_Star_Orbit.csv'):
    """ Analysing the star tracebacks

    :param star_orbit_array:
    :type star_orbit_array: array-like
    :param average_method:
    :type average_method: str
    :param uncertainty_evaluation:
    :type uncertainty_evaluation: str
    :param save:
    :type save: bool
    :param save_path:
    :type save_path: str
    :return:
    :rtype:
    """

    average_star_orbit = np.zeros(np.shape(star_orbit_array)[:-1])
    average_per_star = np.zeros()

    for star in range(np.shape(star_orbit_array)[0]):
        df_average = pd.DataFrame()
        if average_method == 'median':

            df_average[['t', 'X_median', 'Y_median', 'Z_median', 'U_median', 'V_median', 'W_median']] = \
                np.transpose(np.nanmedian(star_orbit_array, axis=2))
        elif average_method == 'mean':
            df_average[['t', 'X_mean', 'Y_mean', 'Z_mean', 'U_mean', 'V_mean', 'W_mean']] = \
                np.transpose(np.nanmean(star_orbit_array, axis=2))
        else:
            raise 'average_method can either "median" or "mean".'

        if uncertainty_evaluation == 'mad':
            df_average[['X_mad', 'Y_mad', 'Z_mad', 'U_mad', 'V_mad', 'W_mad']] = \
                np.transpose(mad(star_orbit_array[1:, :, :], axis=2))
        elif uncertainty_evaluation == 'std':
            df_average[['X_std', 'Y_std', 'Z_std', 'U_std', 'V_std', 'W_std']] = \
                np.transpose(np.nanstd(star_orbit_array[1:, :, :], axis=2))
        else:
            raise 'uncertainty_evaluation can either "mad" or "std".'

        if save:
            df_average.to_csv(str(star) + save_path)

        average_star_orbit[star, :, :] = 0
    return


########################################################################################################################


########################################################################################################################
def sampled_orbitintegration(filepath_or_table, number_of_samples, time_end, time_step, direction='both',
                             sample_method='cluster', average_method='median',
                             reference_orbit_lsr=True, reference_object_pv=None,
                             potential=MWPotential2014):
    """ (Re)sampled Orbit integration of a cluster of stars or individual stars

    ...

    For cluster integration the shape of the returned array is
    (7 (parameters) x number of timesteps x number of samples). The seven parameters are t, X, Y, Z, U, V, W.
    For the integration of single stars the shape of the returned array is
    (number of stars x 7 x number of timesteps x number of samples).

    :param filepath_or_table: table or path to table
    :type filepath_or_table: str, pandas.Dataframe, array-like
    :param number_of_samples: number of times to bootstrap (cluster)
    or sample from the normal distribution (single star)
    :type number_of_samples: int
    :param time_end: absolut of integration time (if -17 Myr -> 17) given as astropy.units.Quantity with time unit.
    If not a units.Quantity it is assumed to be in Myr.
    :type time_end: astropy.units.Quantity, int
    :param time_step: size of timestep as am astropy.units.Quantity. If not a units.Quantity it is assumed to be in Myr.
    :type time_step: astropy.units.Quantity, float, int
    :param direction: 'direction' of integration. Integration 'backward', 'forward' or 'both' (default).
    :type direction: str
    :param sample_method: 'cluster' (default) or 'stellar'. 'cluster' utilises bootstrapping from the cluster members
    to statistically resample an average cluster orbit. 'stellar' integrates each star's orbit, using Monte Carlo-type
    sampling from measurement and measurement uncertainty.
    The number of samples for each method can be set in 'number_of_samples'.
    :type sample_method: str
    :param average_method: 'mean' or 'median' (default) if sample_method is 'cluster'.
    Using either mean or median of each parameter (ra, dec, plx, pmra, pmdec, v_r) of the resampled selection
    of stars for integration per draw.
    :type average_method: str
    :param reference_orbit_lsr: default True, False if LSR is not the reference frame.
    If False, it is necessary to provide position and velocities of the reference frame in the attribute
    'reference_object_pv'.
    :type reference_orbit_lsr: bool
    :param reference_object_pv: positions and velocities of the reference frame. Default is None.
    :type reference_object_pv: 1D array or list
    :param potential: potential in which to integrate the orbit. Choose from galpy.potential and import or define one
    :type potential: galpy.potential

    :return: returns 3- or 4-dimensional array with bootstrapped or sampled and integrated orbits for each timestep
    :rtype: numpy.ndarray
    """

    # convert data to pandas dataframe
    data = read_table_to_df(filepath_or_table)
    number_of_stars = len(data)

    print('-' * 75)
    print(f'Table contains {number_of_stars} stars that are used for orbit calculation.')
    print('-' * 75)

    # create time array for integration and time step size for uniform presentation
    if (isinstance(time_end, unit.Quantity)) & (isinstance(time_step, unit.Quantity)):
        time_unit = time_end.unit
        timerange = unit.Quantity(np.linspace(0, time_end.value, int(time_end / time_step) + 1), time_unit)
    else:
        time_end = unit.Quantity(time_end, unit.Myr)
        time_step = unit.Quantity(time_step, unit.Myr)
        timerange = unit.Quantity(np.linspace(0, time_end.value, int(time_end / time_step) + 1), unit.Myr)

    if direction == 'both':
        arraylength = len(timerange) * 2 - 1
    elif (direction == 'backward') | (direction == 'forward'):
        arraylength = len(timerange)
    else:
        raise 'attribute "direction" needs to be one of the following: "backward", "forward", "both".'

    if sample_method == 'cluster':
        print('... using bootstrapping for tracebacks with method "cluster"')

        # creating (position-time) array in dimensions of
        # dim 0: 7 (for t, x, y, z, u, v, w)
        # dim 1: number of timesteps
        # dim 2: number of sampling repetitions
        pt_array = np.zeros((7, arraylength, number_of_samples))

        n_ra = np.zeros((number_of_samples,)) * unit.deg
        n_dec = np.zeros((number_of_samples,)) * unit.deg
        n_dist = np.zeros((number_of_samples,)) * unit.pc
        n_pmra = np.zeros((number_of_samples,)) * unit.mas / unit.yr
        n_pmdec = np.zeros((number_of_samples,)) * unit.mas / unit.yr
        n_vr = np.zeros((number_of_samples,)) * unit.km / unit.s

        for smpl in range(number_of_samples):

            sampled_data = data.sample(n=number_of_stars, replace=True)

            if average_method == 'mean':
                # averaging over stellar coordinates and velocities
                ra = np.nanmean(sampled_data['ra'])
                dec = np.nanmean(sampled_data['dec'])
                distance = np.nanmean(sampled_data['distance'])
                pmra = np.nanmean(sampled_data['pmra'])
                pmdec = np.nanmean(sampled_data['pmdec'])
                radial_velocity = np.nanmean(sampled_data['radial_velocity'])

            elif average_method == 'median':
                # averaging over stellar coordinates and velocities
                ra = np.nanmedian(sampled_data['ra'])
                dec = np.nanmedian(sampled_data['dec'])
                distance = np.nanmedian(sampled_data['distance'])
                pmra = np.nanmedian(sampled_data['pmra'])
                pmdec = np.nanmedian(sampled_data['pmdec'])
                radial_velocity = np.nanmedian(sampled_data['radial_velocity'])

            else:
                ra, ra_error = 0, 0
                dec, dec_error = 0, 0
                distance, distance_error = 0, 0
                pmra, pmra_error = 0, 0
                pmdec, pmdec_error = 0, 0
                radial_velocity, radial_velocity_error = 0, 0
                raise 'function arg "average_method" is neither set to "mean" or "median".'

            n_ra[smpl] = ra * unit.deg
            n_dec[smpl] = dec * unit.deg
            n_dist[smpl] = distance * unit.pc
            n_pmra[smpl] = pmra * unit.mas / unit.yr
            n_pmdec[smpl] = pmdec * unit.mas / unit.yr
            n_vr[smpl] = radial_velocity * unit.km / unit.s

            # create array with as many random draws as given in number_of_samples
            # n_ra = np.random.normal(ra, ra_error, size=(number_of_samples,)) * unit.deg
            # n_dec = np.random.normal(dec, dec_error, size=(number_of_samples,)) * unit.deg
            # n_dist = np.random.normal(distance, distance_error, size=(number_of_samples,)) * unit.pc
            # n_pmra = np.random.normal(pmra, pmra_error, size=(number_of_samples,)) * unit.mas / unit.yr
            # n_pmdec = np.random.normal(pmdec, pmdec_error, size=(number_of_samples,)) * unit.mas / unit.yr
            # n_vr = np.random.normal(radial_velocity, radial_velocity_error,
            #                         size=(number_of_samples,)) * unit.km / unit.s

        print('... integrating orbits')
        # direction of x is already switched in traceback function.
        # returned x is increasing towards the center
        if direction == 'both':
            # calculating orbits backward
            x_b, y_b, z_b, u_b, v_b, w_b = traceback_stars_radec(
                sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                t=timerange,
                reference_orbit_lsr=reference_orbit_lsr,
                reference_object_pv=reference_object_pv,
                back=True,
                potential=potential)

            # calculating future orbits
            x_f, y_f, z_f, u_f, v_f, w_f = traceback_stars_radec(
                sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                t=timerange,
                reference_orbit_lsr=reference_orbit_lsr,
                reference_object_pv=reference_object_pv,
                back=False,
                potential=potential)

            # creating one time array by sticking backwards array and future array together
            # coordinate_b[:0:-1] reverts the order (0,-15) -> (-15, 0) of the backward array
            # and removes the 0 which is already included in the future array
            x = np.concatenate((x_b[:, :0:-1], x_f), axis=1)
            y = np.concatenate((y_b[:, :0:-1], y_f), axis=1)
            z = np.concatenate((z_b[:, :0:-1], z_f), axis=1)
            u = np.concatenate((u_b[:, :0:-1], u_f), axis=1)
            v = np.concatenate((v_b[:, :0:-1], v_f), axis=1)
            w = np.concatenate((w_b[:, :0:-1], w_f), axis=1)

            t = np.concatenate((timerange[:0:-1] * (-1), timerange))

        elif direction == 'backward':
            # calculating orbits backward
            x_b, y_b, z_b, u_b, v_b, w_b = traceback_stars_radec(
                sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                t=timerange,
                reference_orbit_lsr=reference_orbit_lsr,
                reference_object_pv=reference_object_pv,
                back=True,
                potential=potential)

            x = x_b[:, ::-1]
            y = y_b[:, ::-1]
            z = z_b[:, ::-1]
            u = u_b[:, ::-1]
            v = v_b[:, ::-1]
            w = w_b[:, ::-1]

            t = timerange[::-1] * (-1)

        else:
            # calculating future orbits
            x_f, y_f, z_f, u_f, v_f, w_f = traceback_stars_radec(
                sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                t=timerange,
                reference_orbit_lsr=reference_orbit_lsr,
                reference_object_pv=reference_object_pv,
                back=False,
                potential=potential)

            x = x_f
            y = y_f
            z = z_f
            u = u_f
            v = v_f
            w = w_f

            t = timerange

        print(f'... returning integrated orbits as array with shape\n'
              f' (parameters x timesteps x resamples).'
              f'\n   -> here: (7x{arraylength}x{number_of_samples})\n')

        t = np.reshape(t, (1, len(t))).repeat(number_of_samples, 0)
        # setting columns in array with orbit calculations
        # (x is already corrected for different standard direction)
        pt_array[0, :, :] = np.transpose(t)
        pt_array[1, :, :] = np.transpose(x)
        pt_array[2, :, :] = np.transpose(y)
        pt_array[3, :, :] = np.transpose(z)
        pt_array[4, :, :] = np.transpose(u)
        pt_array[5, :, :] = np.transpose(v)
        pt_array[6, :, :] = np.transpose(w)

    elif sample_method == 'stellar':
        print('... using Monte Carlo-type sampling for star tracebacks with method "stellar"')
        # setting output array with diemsions
        # dim 0: number of stars
        # dim 1: 7 (for t, x, y, z, u, v, w)
        # dim 2: number of timesteps
        # dim 3: number of bootstrapping repetitions
        pt_array = np.zeros((number_of_stars, 7, arraylength, number_of_samples))

        print('... sampling from normal distribution of measurement and measurement uncertainty.')
        print('... integrating orbits')
        # coordinates as Quantities
        for star in range(number_of_stars):

            n_star = data.iloc[star, :]

            n_ra = np.random.normal(n_star['ra'], n_star['ra_error'], size=(number_of_samples,)) * unit.deg
            n_dec = np.random.normal(n_star['dec'], n_star['dec_error'], size=(number_of_samples,)) * unit.deg
            n_dist = np.random.normal(n_star['distance'], n_star['distance_error'], size=(number_of_samples,)) * unit.pc
            n_pmra = np.random.normal(n_star['pmra'], n_star['pmra_error'],
                                      size=(number_of_samples,)) * unit.mas / unit.yr
            n_pmdec = np.random.normal(n_star['pmdec'], n_star['pmdec_error'],
                                       size=(number_of_samples,)) * unit.mas / unit.yr
            n_vr = np.random.normal(n_star['radial_velocity'], n_star['radial_velocity_error'],
                                    size=(number_of_samples,)) * unit.km / unit.s

            if direction == 'both':
                # calculating orbits backward
                x_b, y_b, z_b, u_b, v_b, w_b = traceback_stars_radec(
                    sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                    t=timerange,
                    reference_orbit_lsr=reference_orbit_lsr,
                    reference_object_pv=reference_object_pv,
                    back=True,
                    potential=potential)

                # calculating future orbits
                x_f, y_f, z_f, u_f, v_f, w_f = traceback_stars_radec(
                    sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                    t=timerange,
                    reference_orbit_lsr=reference_orbit_lsr,
                    reference_object_pv=reference_object_pv,
                    back=False,
                    potential=potential)

                # creating one time array by sticking backwards array and future array together
                # coordinate_b[:0:-1] reverts the order (0,-15) -> (-15, 0) of the backward array
                # and removes the 0 which is already included in the future array
                x = np.concatenate((x_b[:, :0:-1], x_f), axis=1)
                y = np.concatenate((y_b[:, :0:-1], y_f), axis=1)
                z = np.concatenate((z_b[:, :0:-1], z_f), axis=1)
                u = np.concatenate((u_b[:, :0:-1], u_f), axis=1)
                v = np.concatenate((v_b[:, :0:-1], v_f), axis=1)
                w = np.concatenate((w_b[:, :0:-1], w_f), axis=1)

                t = np.concatenate((timerange[:0:-1] * (-1), timerange))

            elif direction == 'backward':
                # calculating orbits backward
                x_b, y_b, z_b, u_b, v_b, w_b = traceback_stars_radec(
                    sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                    t=timerange,
                    reference_orbit_lsr=reference_orbit_lsr,
                    reference_object_pv=reference_object_pv,
                    back=True,
                    potential=potential)

                x = x_b[:, ::-1]
                y = y_b[:, ::-1]
                z = z_b[:, ::-1]
                u = u_b[:, ::-1]
                v = v_b[:, ::-1]
                w = w_b[:, ::-1]

                t = timerange[::-1] * (-1)

            else:
                # calculating future orbits
                x_f, y_f, z_f, u_f, v_f, w_f = traceback_stars_radec(
                    sky_object_pv=[n_ra, n_dec, n_dist, n_pmra, n_pmdec, n_vr],
                    t=timerange,
                    reference_orbit_lsr=reference_orbit_lsr,
                    reference_object_pv=reference_object_pv,
                    back=False,
                    potential=potential)

                x = x_f
                y = y_f
                z = z_f
                u = u_f
                v = v_f
                w = w_f

                t = timerange

            t = np.reshape(t, (1, len(t))).repeat(number_of_samples, 0)
            # setting columns in array with orbit calculations
            # (x is already corrected for different standard direction)
            pt_array[star, 0, :, :] = np.transpose(t)
            pt_array[star, 1, :, :] = np.transpose(x)
            pt_array[star, 2, :, :] = np.transpose(y)
            pt_array[star, 3, :, :] = np.transpose(z)
            pt_array[star, 4, :, :] = np.transpose(u)
            pt_array[star, 5, :, :] = np.transpose(v)
            pt_array[star, 6, :, :] = np.transpose(w)

        print(f'... returning integrated stellar orbits as array with shape\n'
              f' (number of stars x parameters x timesteps x samples).'
              f'\n   -> here: ({number_of_stars} x 7 x {arraylength} x {number_of_samples})\n')

    else:
        pt_array = np.zeros((1,))
        raise 'sample method must be either "stellar" or "cluster".'

    return pt_array


########################################################################################################################


########################################################################################################################
def traceback_stars_radec(sky_object_pv, t, reference_orbit_lsr=True, reference_object_pv=None, back=True,
                          potential=MWPotential2014):
    """
    code to calculate the traceback of one or more sky_objects initialised by heliocentric equatorial coordinates
    (like in *GAIA*). Output is in Cartesian coordinates with the center of the coordinate system being the
    reference frame given.


    :param sky_object_pv: position and velocity of traceable object(s), in the form
    [ra, dec, distance, pmra, pmdec, radial velocity]. Single values or array of values for each coordinate.
    Can but do not have to be astropy.units.Quantity.
    :type sky_object_pv: list[int, float, Quantity]
    :param t: time value(s) at which to evaluate the orbit (0 to t_n)
    :type t: astropy.units.Quantity
    :param reference_orbit_lsr: default *True* for LSR as reference frame
    (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s.
    If *False*, [ra, dec, distance, pmra, pmdec, vr] needs to be passed to the reference_object_pv parameter
    :type reference_orbit_lsr: bool
    :param reference_object_pv: position and velocity of reference object if reference_orbit is *False*
    :type reference_object_pv: list
    :param back: default *True*, integrates backward in time (flips time sequence and velocities).
    If back is *False* integrates forward in time.
    :type back: bool
    :param potential: default MWPotential2014, any other galpy potential can be passed
    (https://docs.galpy.org/en/latest/potential.html)
    :type potential: galpy.potential

    :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s.
    For each coordinate an array is returned with shape (len(t), len(sky_object_pv[any]):
    for each timestep and sky object integrated positions and velocities are returned.
    :rtype: float, array
    """

    gc = Galactocentric()

    ra_so = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
    dec_so = unit.Quantity(sky_object_pv[1], unit.deg, copy=False)
    distance_so = unit.Quantity(sky_object_pv[2], unit.kpc, copy=False)
    pmra_so = unit.Quantity(sky_object_pv[3], unit.mas / unit.yr, copy=False)
    pmdec_so = unit.Quantity(sky_object_pv[4], unit.mas / unit.yr, copy=False)
    radialvelocity_so = unit.Quantity(sky_object_pv[5], unit.km / unit.s, copy=False)

    t = unit.Quantity(t, unit.Myr, copy=False)

    # reference frame or observer's orbit
    if reference_orbit_lsr:
        # reference frame is the LSR at (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s
        reference_orbit = Orbit(
            [0 * unit.deg, 0 * unit.deg, 0 * unit.kpc, -LSR.v_bary.d_x, -LSR.v_bary.d_y, -LSR.v_bary.d_z],
            radec=True,
            lb=False,
            uvw=True,
            ro=gc.galcen_distance,
            zo=0,
            vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
            solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    elif (reference_object_pv is not None) & (reference_orbit_lsr is False):
        ra_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        dec_ro = unit.Quantity(sky_object_pv[1], unit.deg, copy=False)
        distance_ro = unit.Quantity(sky_object_pv[2], unit.kpc, copy=False)
        pmra_ro = unit.Quantity(sky_object_pv[3], unit.mas / unit.yr, copy=False)
        pmdec_ro = unit.Quantity(sky_object_pv[4], unit.mas / unit.yr, copy=False)
        radialvelocity_ro = unit.Quantity(sky_object_pv[5], unit.km / unit.s, copy=False)

        # custom reference frame
        reference_orbit = Orbit([ra_ro, dec_ro, distance_ro, pmra_ro, pmdec_ro, radialvelocity_ro],
                                radec=True,
                                lb=False,
                                uvw=False,
                                ro=gc.galcen_distance,
                                zo=0,
                                vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                                solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    elif (reference_object_pv is None) & (reference_orbit_lsr is False):
        raise 'reference_orbit_lsr is set to False but no reference frame is provided in the ' \
              'parameter reference_object_pv.'

    elif (reference_object_pv is not None) & (reference_orbit_lsr is True):
        del reference_object_pv

    else:
        raise 'Reference orbit not defined. Set either reference_orbit_lsr to True or set it to' \
              'False and provide coordinates for a reference frame in reference_object_pv.'

    # orbit of sky object(s)
    skyobject_orbit = Orbit([ra_so, dec_so, distance_so, pmra_so, pmdec_so, radialvelocity_so],
                            radec=True,
                            lb=False,
                            uvw=False,
                            ro=gc.galcen_distance,
                            zo=gc.z_sun,
                            vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                            solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    # if using 'back=True' function will return backwards integration
    # if using 'back=False' function will integrate into the future
    if back:
        skyobject_orbit.flip(inplace=True)
        reference_orbit.flip(inplace=True)

    # integration with chosen potential
    skyobject_orbit.integrate(t=t, pot=potential)
    reference_orbit.integrate(t=t, pot=potential)

    # setting output arrays as 'orbit of stars - orbit of observer'
    # at the given times
    x = skyobject_orbit.x(t) - reference_orbit.x(t)
    y = skyobject_orbit.y(t) - reference_orbit.y(t)
    z = skyobject_orbit.z(t) - reference_orbit.z(t)

    if back:
        u = skyobject_orbit.vx(t) - reference_orbit.vx(t)
        v = - (skyobject_orbit.vy(t) - reference_orbit.vy(t))
        w = - (skyobject_orbit.vz(t) - reference_orbit.vz(t))
    else:
        u = - (skyobject_orbit.vx(t) - reference_orbit.vx(t))
        v = skyobject_orbit.vy(t) - reference_orbit.vy(t)
        w = skyobject_orbit.vz(t) - reference_orbit.vz(t)

    # return coordinates as array in pc
    return x * (-1e3), y * 1e3, z * 1e3, u, v, w

########################################################################################################################
