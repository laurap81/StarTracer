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


########################################################################################################################
def mad(array, axis=1):
    """
    function calculates the median absolut deviation of a 1D or 2D array along either axis.
    :param array: Input array of which to calculate the median absolut deviation of.
    :type array: array
    :param axis: 0 or 1 for a 2D array. 0 for a 1D array. Calculates mean and median along this axis.
    default is axis=1 (rows)
    :type axis: int
    :return: median absolut deviation of array
    :rtype: array, float
    """
    if axis == 1:
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(array, np.nanmean(array, axis=1)[:, None])), axis=1)
    elif axis == 0:
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(array, np.nanmean(array, axis=0))), axis=0)
    else:
        raise 'dimension not implemented'
    return median_absolut_deviation
########################################################################################################################


########################################################################################################################
def skycoord_from_table(path_to_file):
    """
    create a astropy.coordinates.SkyCoord from some input table. if no column found that is called 'distance',
    parallax is automatically converted to distance
    :param path_to_file: path to table file
    :type path_to_file: str
    :return: SkyCoord object of
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
    # one dimension needs to match the number of columnns (6D coordinates + errors -> 12 columns)
    elif isinstance(filepath_or_table, np.ndarray) & ((np.shape(filepath_or_table)[0] == len(column_names))
                                                      | (np.shape(filepath_or_table)[1] == len(column_names))):
        df = pd.DataFrame(filepath_or_table,
                          columns=column_names)
    else:
        raise 'Data input must be either a path to a table, a pandas dataframe or numpy ndarray.'

    return df
########################################################################################################################


########################################################################################################################
def mc_sampling_orbits(filepath_or_table, number_of_samples, time_end, time_step, direction='both',
                       sample_method='cluster', average_method='mean',
                       reference_orbit_lsr=True, reference_object_pv=None,
                       potential=MWPotential2014):
    """

    :param filepath_or_table: table or path to table
    :type filepath_or_table: str, pandas.Dataframe, array
    :param number_of_samples: number of times to sample from the normal distribution
    :type number_of_samples: int
    :param time_end: absolut of integration time (to -17 Myr -> 17) given as astropy.units.Quantity with time unit.
    If not a units.Quantity it is assumed to be in Myr.
    :type time_end: astropy.units.Quantity, int
    :param time_step: size of timestep as am astropy.units.Quantity. If not a units.Quantity it is assumed to be in Myr.
    :type time_step: astropy.units.Quantity, float, int
    :param direction: 'direction' of integration. Integration 'backward', 'forward' or 'both'. Default is 'both'.
    :type direction: str
    :param sample_method: 'cluster' or 'stellar'. 'cluster' averages over the input data and calculates one orbit,
    statistically sampled, from all stellar orbits. 'stellar' integrates each star's orbit,
    statistically sampled from measurement and error, as many times as defined in 'number_of_samples'.
    :type sample_method: str, (default is 'cluster')
    :param average_method: 'mean' or 'median' if sample_method='cluster'.
    'mean' draws from a normal distribution of the mean and standard error of all stars,
    'median' draws from a normal distribution of the median and median absolut deviation of all stars.
    :type average_method: str, (default is 'mean')
    :param reference_orbit_lsr: default True, False if LSR is not the reference frame.
    If False, it is necessary to provide position and velocities of the reference frame in the attribute
    'reference_object_pv'.
    :type reference_orbit_lsr: bool
    :param reference_object_pv: positions and velocities of the reference frame. Default is None.
    :type reference_object_pv: 1D array or list
    :param potential: potential in which to integrate the orbit. Choose from galpy.potential and import or define one
    :type potential: galpy.potential

    :return: returns 3D array with sampled and integrated orbits for each timestep
    :rtype: numpy.ndarray
    """
    # convert data to pandas dataframe
    data = read_table_to_df(filepath_or_table)

    print(f'Table contains {len(data)} stars that are used to calculate orbits.')
    print('-' * 100)

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
        arraylength = len(timerange) - 1
    else:
        raise 'attribute "direction" needs to be one of the following: "backward", "forward", "both".'

    if sample_method == 'cluster':
        print('... using averaging sample method ("cluster")')

        if average_method == 'mean':
            print('... using the "mean" and "standard error" as averaging method')
            # averaging over stellar coordinates and velocities
            ra, ra_error = np.nanmean(data['ra']), np.nanstd(data['ra'].values) / np.sqrt(len(data))
            dec, dec_error = np.nanmean(data['dec']), np.nanstd(data['dec'].values) / np.sqrt(len(data))
            distance, distance_error = np.nanmean(data['distance']), np.nanstd(data['distance'].values) / np.sqrt(
                len(data))
            pmra, pmra_error = np.nanmean(data['pmra']), np.nanstd(data['pmra'].values) / np.sqrt(len(data))
            pmdec, pmdec_error = np.nanmean(data['pmdec']), np.nanstd(data['pmdec'].values) / np.sqrt(len(data))
            radial_velocity, radial_velocity_error = np.nanmean(data['radial_velocity']), \
                np.nanstd(data['radial_velocity'].values) / np.sqrt(len(data))

        elif average_method == 'median':
            print('... using the "median" and "median absolut deviation" as averaging method')
            # averaging over stellar coordinates and velocities
            ra, ra_error = np.nanmedian(data['ra']), mad(data['ra'].values)
            dec, dec_error = np.nanmedian(data['dec']), mad(data['dec'].values)
            distance, distance_error = np.nanmedian(data['distance']), mad(data['distance'].values)
            pmra, pmra_error = np.nanmedian(data['pmra']), mad(data['pmra'].values)
            pmdec, pmdec_error = np.nanmedian(data['pmdec']), mad(data['pmdec'].values)
            radial_velocity, radial_velocity_error = np.nanmedian(data['radial_velocity']), \
                mad(data['radial_velocity'].values)

        else:
            ra, ra_error = 0, 0
            dec, dec_error = 0, 0
            distance, distance_error = 0, 0
            pmra, pmra_error = 0, 0
            pmdec, pmdec_error = 0, 0
            radial_velocity, radial_velocity_error = 0, 0
            raise 'function arg "average_method" is neither set to "mean" or "median".'

        # creating (position-time) array in dimensions of
        # dim 0: 4 (for t, x, y, z)
        # dim 1: number of timesteps
        # dim 2: number of bootstrapping repetitions
        pt_array = np.zeros((7, arraylength, number_of_samples))
        print('... Monte Carlo type sampling from standard errors')
        for nn in range(number_of_samples):
            n_ra = np.random.normal(ra, ra_error) * unit.deg
            n_dec = np.random.normal(dec, dec_error) * unit.deg
            n_dist = np.random.normal(distance, distance_error) * unit.pc
            n_pmra = np.random.normal(pmra, pmra_error) * unit.mas / unit.yr
            n_pmdec = np.random.normal(pmdec, pmdec_error) * unit.mas / unit.yr
            n_vr = np.random.normal(radial_velocity, radial_velocity_error) * unit.km / unit.s

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
                x = np.concatenate((x_b[:0:-1], x_f))
                y = np.concatenate((y_b[:0:-1], y_f))
                z = np.concatenate((z_b[:0:-1], z_f))
                u = np.concatenate((u_b[:0:-1], u_f))
                v = np.concatenate((v_b[:0:-1], v_f))
                w = np.concatenate((w_b[:0:-1], w_f))

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

                x = x_b[::-1]
                y = y_b[::-1]
                z = z_b[::-1]
                u = u_b[::-1]
                v = v_b[::-1]
                w = w_b[::-1]

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

            # setting columns in array with orbit calculations plus reverting x
            # (correct for different standard direction in x in galpy)
            pt_array[0, :, nn] = t
            pt_array[1, :, nn] = x
            pt_array[2, :, nn] = y
            pt_array[3, :, nn] = z
            pt_array[4, :, nn] = u
            pt_array[5, :, nn] = v
            pt_array[6, :, nn] = w

            # if nn % 100 == 0:
            #     print(nn)

        print('... integrated orbits')
        print(f'... returning integrated orbits (7x{arraylength}x{number_of_samples} array)')

    elif sample_method == 'stellar':
        print('... using single stars for traceback')
        pt_array = np.zeros((7, arraylength, number_of_samples))

        print('... sampling from measurement and measurement uncertainty.')
        # coordinates as Quantities
        for star in range(len(data)):
            for nn in range(number_of_samples):

                n_ra = np.random.normal(data['ra'], data['ra_error']) * unit.deg
                n_dec = np.random.normal(data['dec'], data['dec_error']) * unit.deg
                n_dist = np.random.normal(data['distance'], data['distance_error']) * unit.pc
                n_pmra = np.random.normal(data['pmra'], data['pmra_error']) * unit.mas / unit.yr
                n_pmdec = np.random.normal(data['pmdec'], data['pmdec_error']) * unit.mas / unit.yr
                n_vr = np.random.normal(data['radial_velocity'], data['radial_velocity_error']) * unit.km / unit.s
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
                    x = np.concatenate((x_b[:0:-1], x_f))
                    y = np.concatenate((y_b[:0:-1], y_f))
                    z = np.concatenate((z_b[:0:-1], z_f))
                    u = np.concatenate((u_b[:0:-1], u_f))
                    v = np.concatenate((v_b[:0:-1], v_f))
                    w = np.concatenate((w_b[:0:-1], w_f))

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

                    x = x_b[::-1]
                    y = y_b[::-1]
                    z = z_b[::-1]
                    u = u_b[::-1]
                    v = v_b[::-1]
                    w = w_b[::-1]

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

                # setting columns in array with orbit calculations plus reverting x
                # (correct for different standard direction in x in galpy)
                pt_array[0, :, nn] = t
                pt_array[1, :, nn] = x
                pt_array[2, :, nn] = y
                pt_array[3, :, nn] = z
                pt_array[4, :, nn] = u
                pt_array[5, :, nn] = v
                pt_array[6, :, nn] = w

        print('... integrated orbits')
        print(f'... returning integrated orbits (7x{arraylength}x{number_of_samples} array)')

    else:
        pt_array = np.zeros((7, arraylength, number_of_samples))
        raise 'sample method must be either "stellar" or "cluster".'

    return pt_array
########################################################################################################################


########################################################################################################################
def bootstrapping_orbits(filepath_or_table, number_of_samples, time_end, time_step, direction='both',
                         sample_method='cluster', average_method='mean',
                         reference_orbit_lsr=True, reference_object_pv=None,
                         potential=MWPotential2014):
    """

    :param filepath_or_table: table or path to table
    :type filepath_or_table: str, pandas.Dataframe, array
    :param number_of_samples: number of times to sample from the normal distribution
    :type number_of_samples: int
    :param time_end: absolut of integration time (to -17 Myr -> 17) given as astropy.units.Quantity with time unit.
    If not a units.Quantity it is assumed to be in Myr.
    :type time_end: astropy.units.Quantity, int
    :param time_step: size of timestep as am astropy.units.Quantity. If not a units.Quantity it is assumed to be in Myr.
    :type time_step: astropy.units.Quantity, float, int
    :param direction: 'direction' of integration. Integration 'backward', 'forward' or 'both'. Default is 'both'.
    :type direction: str
    :param sample_method: 'cluster' or 'stellar'. 'cluster' averages over the input data and calculates one orbit,
    statistically sampled, from all stellar orbits. 'stellar' integrates each star's orbit,
    statistically sampled from measurement and error, as many times as defined in 'number_of_samples'.
    :type sample_method: str, (default is 'cluster')
    :param average_method: 'mean' or 'median' if sample_method='cluster'.
    'mean' draws from a normal distribution of the mean and standard error of all stars,
    'median' draws from a normal distribution of the median and median absolut deviation of all stars.
    :type average_method: str, (default is 'mean')
    :param reference_orbit_lsr: default True, False if LSR is not the reference frame.
    If False, it is necessary to provide position and velocities of the reference frame in the attribute
    'reference_object_pv'.
    :type reference_orbit_lsr: bool
    :param reference_object_pv: positions and velocities of the reference frame. Default is None.
    :type reference_object_pv: 1D array or list
    :param potential: potential in which to integrate the orbit. Choose from galpy.potential and import or define one
    :type potential: galpy.potential

    :return: returns 3D array with sampled and integrated orbits for each timestep
    :rtype: numpy.ndarray
    """
    # convert data to pandas dataframe
    data = read_table_to_df(filepath_or_table)

    print(f'Table contains {len(data)} stars that are used to calculate orbits.')
    print('-' * 100)

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
        arraylength = len(timerange) - 1
    else:
        raise 'attribute "direction" needs to be one of the following: "backward", "forward", "both".'

    if sample_method == 'cluster':
        print('... using averaging sample method ("cluster")')

        if average_method == 'mean':
            print('... using the "mean" and "standard error" as averaging method')
            # averaging over stellar coordinates and velocities
            ra, ra_error = np.nanmean(data['ra']), np.nanstd(data['ra'].values) / np.sqrt(len(data))
            dec, dec_error = np.nanmean(data['dec']), np.nanstd(data['dec'].values) / np.sqrt(len(data))
            distance, distance_error = np.nanmean(data['distance']), np.nanstd(data['distance'].values) / np.sqrt(
                len(data))
            pmra, pmra_error = np.nanmean(data['pmra']), np.nanstd(data['pmra'].values) / np.sqrt(len(data))
            pmdec, pmdec_error = np.nanmean(data['pmdec']), np.nanstd(data['pmdec'].values) / np.sqrt(len(data))
            radial_velocity, radial_velocity_error = np.nanmean(data['radial_velocity']), \
                np.nanstd(data['radial_velocity'].values) / np.sqrt(len(data))

        elif average_method == 'median':
            print('... using the "median" and "median absolut deviation" as averaging method')
            # averaging over stellar coordinates and velocities
            ra, ra_error = np.nanmedian(data['ra']), mad(data['ra'].values)
            dec, dec_error = np.nanmedian(data['dec']), mad(data['dec'].values)
            distance, distance_error = np.nanmedian(data['distance']), mad(data['distance'].values)
            pmra, pmra_error = np.nanmedian(data['pmra']), mad(data['pmra'].values)
            pmdec, pmdec_error = np.nanmedian(data['pmdec']), mad(data['pmdec'].values)
            radial_velocity, radial_velocity_error = np.nanmedian(data['radial_velocity']), \
                mad(data['radial_velocity'].values)

        else:
            ra, ra_error = 0, 0
            dec, dec_error = 0, 0
            distance, distance_error = 0, 0
            pmra, pmra_error = 0, 0
            pmdec, pmdec_error = 0, 0
            radial_velocity, radial_velocity_error = 0, 0
            raise 'function arg "average_method" is neither set to "mean" or "median".'

        # creating (position-time) array in dimensions of
        # dim 0: 4 (for t, x, y, z)
        # dim 1: number of timesteps
        # dim 2: number of bootstrapping repetitions
        pt_array = np.zeros((7, arraylength, number_of_samples))
        print('... Monte Carlo type sampling from standard errors')
        for nn in range(number_of_samples):
            n_ra = np.random.normal(ra, ra_error) * unit.deg
            n_dec = np.random.normal(dec, dec_error) * unit.deg
            n_dist = np.random.normal(distance, distance_error) * unit.pc
            n_pmra = np.random.normal(pmra, pmra_error) * unit.mas / unit.yr
            n_pmdec = np.random.normal(pmdec, pmdec_error) * unit.mas / unit.yr
            n_vr = np.random.normal(radial_velocity, radial_velocity_error) * unit.km / unit.s

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
                x = np.concatenate((x_b[:0:-1], x_f))
                y = np.concatenate((y_b[:0:-1], y_f))
                z = np.concatenate((z_b[:0:-1], z_f))
                u = np.concatenate((u_b[:0:-1], u_f))
                v = np.concatenate((v_b[:0:-1], v_f))
                w = np.concatenate((w_b[:0:-1], w_f))

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

                x = x_b[::-1]
                y = y_b[::-1]
                z = z_b[::-1]
                u = u_b[::-1]
                v = v_b[::-1]
                w = w_b[::-1]

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

            # setting columns in array with orbit calculations plus reverting x
            # (correct for different standard direction in x in galpy)
            pt_array[0, :, nn] = t
            pt_array[1, :, nn] = x
            pt_array[2, :, nn] = y
            pt_array[3, :, nn] = z
            pt_array[4, :, nn] = u
            pt_array[5, :, nn] = v
            pt_array[6, :, nn] = w

            # if nn % 100 == 0:
            #     print(nn)

        print('... integrated orbits')
        print(f'... returning integrated orbits (7x{arraylength}x{number_of_samples} array)')

    elif sample_method == 'stellar':
        print('... using single stars for traceback')
        pt_array = np.zeros((7, arraylength, number_of_samples))

        print('... sampling from measurement and measurement uncertainty.')
        # coordinates as Quantities
        for star in range(len(data)):
            for nn in range(number_of_samples):

                n_ra = np.random.normal(data['ra'], data['ra_error']) * unit.deg
                n_dec = np.random.normal(data['dec'], data['dec_error']) * unit.deg
                n_dist = np.random.normal(data['distance'], data['distance_error']) * unit.pc
                n_pmra = np.random.normal(data['pmra'], data['pmra_error']) * unit.mas / unit.yr
                n_pmdec = np.random.normal(data['pmdec'], data['pmdec_error']) * unit.mas / unit.yr
                n_vr = np.random.normal(data['radial_velocity'], data['radial_velocity_error']) * unit.km / unit.s
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
                    x = np.concatenate((x_b[:0:-1], x_f))
                    y = np.concatenate((y_b[:0:-1], y_f))
                    z = np.concatenate((z_b[:0:-1], z_f))
                    u = np.concatenate((u_b[:0:-1], u_f))
                    v = np.concatenate((v_b[:0:-1], v_f))
                    w = np.concatenate((w_b[:0:-1], w_f))

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

                    x = x_b[::-1]
                    y = y_b[::-1]
                    z = z_b[::-1]
                    u = u_b[::-1]
                    v = v_b[::-1]
                    w = w_b[::-1]

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

                # setting columns in array with orbit calculations plus reverting x
                # (correct for different standard direction in x in galpy)
                pt_array[0, :, nn] = t
                pt_array[1, :, nn] = x
                pt_array[2, :, nn] = y
                pt_array[3, :, nn] = z
                pt_array[4, :, nn] = u
                pt_array[5, :, nn] = v
                pt_array[6, :, nn] = w

        print('... integrated orbits')
        print(f'... returning integrated orbits (7x{arraylength}x{number_of_samples} array)')

    else:
        pt_array = np.zeros((7, arraylength, number_of_samples))
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
    :type reference_object_lsr: bool
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

    if (reference_object_pv is None) & (reference_orbit_lsr is False):
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


########################################################################################################################
def traceback_stars_galactic(sky_object_pv: list, t: np.ndarray or list, reference_orbit_lsr=True,
                             reference_object_pv=None, back=True, potential=MWPotential2014):
    """
    code to calculate the traceback of one or more sky_objects initialised by heliocentric galactic coordinates.
    Output is in Cartesian coordinates with the center of the coordinate system being the reference frame given.


    :param list[int, float, Quantity] sky_object_pv: position and velocity of traceable object(s),
    in the form [glon, glat, distance, pmll, pmbb, radial velocity]. Single values or array of values for each coordinate.
    Can but do not have to be astropy.units.Quantity.
    :param any t: time value(s) at which to evaluate the orbit (0 to t_n)
    :param bool reference_orbit_lsr: default *True* for LSR as reference frame
    (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s.
    If *False*, [glon, glat, distance, pmll, pmbb, radial velocity] needs to be passed to the reference_object_pv
    parameter.
    :param list reference_object_pv: position and velocity of reference object if reference_orbit is *False*
    :param bool back: default *True*, integrates backward in time (flips time sequence and velocities).
    If back is *False* integrates forward in time.
    :param galpy.potential potential: default MWPotential2014, any other galpy potential can be passed
    (https://docs.galpy.org/en/latest/potential.html)

    :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s.
    For each coordinate an array is returned with shape (len(t), len(sky_object_pv[any]):
    for each timestep and sky object integrated positions and velocities are returned.
    :rtype: float, array
    """

    gc = Galactocentric()

    glon_so = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
    glat_so = unit.Quantity(sky_object_pv[1], unit.deg, copy=False)
    distance_so = unit.Quantity(sky_object_pv[2], unit.kpc, copy=False)
    pml_so = unit.Quantity(sky_object_pv[3], unit.mas / unit.yr, copy=False)
    pmb_so = unit.Quantity(sky_object_pv[4], unit.mas / unit.yr, copy=False)
    radialvelocity_so = unit.Quantity(sky_object_pv[5], unit.km / unit.s, copy=False)

    t = unit.Quantity(t, unit.Myr, copy=False)

    # reference frame or observer's orbit
    if reference_orbit_lsr:
        # reference frame is the LSR at (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s
        reference_orbit = Orbit(
            [0 * unit.deg, 0 * unit.deg, 0 * unit.kpc, -LSR.v_bary.d_x, -LSR.v_bary.d_y, -LSR.v_bary.d_z],
            radec=False,
            lb=True,
            uvw=True,
            ro=gc.galcen_distance,
            zo=0,
            vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
            solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    elif (reference_object_pv is not None) & (reference_orbit_lsr is False):
        glon_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        glat_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        distance_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        pml_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        pmb_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
        radialvelocity_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)

        # custom reference frame
        reference_orbit = Orbit([glon_ro, glat_ro, distance_ro, pml_ro, pmb_ro, radialvelocity_ro],
                                radec=False,
                                lb=True,
                                uvw=False,
                                ro=gc.galcen_distance,
                                zo=0,
                                vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                                solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    if (reference_object_pv is None) & (reference_orbit_lsr is False):
        raise 'reference_orbit_lsr is set to False but no reference frame is provided in the ' \
              'parameter reference_object_pv.'

    elif (reference_object_pv is not None) & (reference_orbit_lsr is True):
        del reference_object_pv

    else:
        raise 'Reference orbit not defined. Set either reference_orbit_lsr to True or set it to' \
              'False and provide coordinates for a reference frame in reference_object_pv.'

    # orbit of sky object(s)
    skyobject_orbit = Orbit([glon_so, glat_so, distance_so, pml_so, pmb_so, radialvelocity_so],
                            radec=False,
                            lb=True,
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
