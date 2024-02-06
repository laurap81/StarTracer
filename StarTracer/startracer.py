from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from astropy.coordinates import Galactocentric, LSR, Distance, SkyCoord
import astropy.units as unit
from astropy.table import Table, QTable

import pandas as pd
import numpy as np

# todo: mc type sampling for distance from parallax conversion
# todo: include this in the documentation

# todo: fix single star sampling
# todo: if time left, include plot function


class Cluster:
    """Samples and integrates orbits from the input data.

    :param input_data: input file for stellar cluster members that is read by the ``read_table_to_df()`` function and
        converted to a :class:`pandas.DataFrame`. If data is provided as a numpy.ndarray dimensions must be
        (12xN) or (Nx12) with the 12 parameters being ['ra', 'ra_error', 'dec', 'dec_error', 'distance',
        'distance_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error']
        in exactly this order.
    :type input_data: str, pandas.DataFrame, astropy.table.Table, numpy.ndarray

    |

    Examples
    ========

    Initialising the cluster and accessing the DataFrame with the attribute ``.data``.

    .. code-block::

        >>> from astropy.table import Table
        >>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
        >>> cluster = Cluster(cluster_data)
        >>> cluster.data.head()
                   ra        dec  ...  pmdec_error  radial_velocity_error
        0  285.437166 -36.976161  ...     0.065070               0.081022
        1  285.063548 -36.843633  ...     0.153469               0.318714
        2  285.133199 -37.037393  ...     0.054875               0.183739
        3  285.191013 -36.617811  ...     0.120342               0.696206
        4  285.039615 -36.946535  ...     0.209726               0.046242
        <BLANKLINE>
        [5 rows x 12 columns]
    """

    def __init__(self, input_data):
        """Constructor method
        """
        self.data = read_table_to_df(input_data)

    def sample_orbit(self, time_end, time_step, number_of_samples=1000, direction='both', average_method='median',
                     reference_orbit_lsr=True, reference_object_pv=None, potential=MWPotential2014,
                     print_out=False):
        """Resamples orbit integration for a cluster of stars.

        Bootstraps N-times over the stellar cluster members with replacement. For details see pandas.DataFrame.sample().
        Averages of the positions and velocities of each of the N samples are integrated and collected in a
        3-dimensional numpy.ndarray.
        The shape of the returned array is (7 (parameters) x number of timesteps x number of samples).
        The seven parameters are t, X, Y, Z, U, V, W.

        :param time_end: Value of integration time given as astropy.units.Quantity with time unit.
            If not given as astropy.units.Quantity it is assumed to be in Myr. Cannot be 0 (zero).
        :type time_end: astropy.units.Quantity, int
        :param time_step: Size of timestep as am astropy.units.Quantity. If not a units.Quantity it is assumed to be
            in Myr. Needs to be smaller than time_end and cannot be 0 (zero).
        :type time_step: astropy.units.Quantity, float, int
        :param number_of_samples: Number of times to bootstrap from cluster members. Defaults to 1000.
        :type number_of_samples: int, float, optional
        :param direction: 'direction' of integration. Integration 'backward', 'forward' or 'both' (default).
        :type direction: str, optional
        :param average_method: 'mean' or 'median' (default) Using either mean or median of each parameter
            (ra, dec, plx, pmra, pmdec, v_r) of the resampled selection of stars for integration per draw.
        :type average_method: str, optional
        :param reference_orbit_lsr: default True, False if LSR is not the reference frame.
            If False, it is necessary to provide position and velocities of the reference frame in the attribute
            'reference_object_pv'.
        :type reference_orbit_lsr: bool, optional
        :param reference_object_pv: positions and velocities of the reference frame. Default is None.
        :type reference_object_pv: 1D array or list, optional
        :param potential: potential in which to integrate the orbit. Choose from galpy.potential and import or define
            a personalised potential
        :type potential: galpy.potential, optional
        :param print_out: If True, prints information and updates on integration. Defaults to False.
        :type print_out: bool, optional

        :return: returns 3-dimensional numpy.ndarray with bootstrapped and integrated orbits for each timestep
            and parameter (t, X, Y, Z, U, V, W).
        :rtype: SampledCluster

        :raises ValueError: If either of time_end or time_step is 0 (zero) or time_step is
            greater than or equal to time_end.
        :raises ValueError: If direction is not among 'backward', 'forward', or 'both'.
        :raises ValueError: If average_method is neither 'mean' nor 'median'.

        |

        Examples
        ========

        Initialising the cluster and applying the method to integrate the sampled orbits. When integration direction
        is set to 'both', the resulting array has twice the
        | number of timesteps - 1: t = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5). The default number of samples is 1000.

        .. code-block::

            >>> from astropy.table import Table
            >>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
            >>> cluster = Cluster(cluster_data)

            >>> data_cluster_both = cluster.sample_orbit(5, 1, number_of_samples=1000, direction='both').get_data()
            >>> np.shape(data_cluster_both)
            (7, 11, 1000)
            >>> data_cluster_back = cluster.sample_orbit(5, 1, direction='backward').get_data()
            >>> np.shape(data_cluster_back)
            (7, 6, 1000)
        """

        data = self.data
        number_of_stars = len(data.index)

        time_end = np.abs(time_end)
        time_step = np.abs(time_step)

        if time_end == 0:
            raise ValueError('time_end cannot be 0.')
        elif time_step == 0:
            raise ValueError('time_step cannot be 0.')
        elif time_step >= time_end:
            raise ValueError('time_step cannot be greater than or equal time_end.')

        if print_out:
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
            array_length = len(timerange) * 2 - 1
        elif (direction == 'backward') | (direction == 'forward'):
            array_length = len(timerange)
        else:
            raise ValueError(f'direction={direction} is not valid.'
                             f'Set it to one of the following: "backward", "forward", "both" (default).')

        if print_out:
            print('... using bootstrapping for tracebacks.')
        # creating (position-time) array in dimensions of
        # dim 0: 7 (for t, x, y, z, u, v, w)
        # dim 1: number of timesteps
        # dim 2: number of sampling repetitions
        number_of_samples = int(number_of_samples)
        pt_array = np.zeros((7, array_length, number_of_samples))

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
                raise ValueError(f'average_method={average_method} is not valid. Set it to "mean" or'
                                 f'"median" (default).')

            n_ra[smpl] = ra * unit.deg
            n_dec[smpl] = dec * unit.deg
            n_dist[smpl] = distance * unit.pc
            n_pmra[smpl] = pmra * unit.mas / unit.yr
            n_pmdec[smpl] = pmdec * unit.mas / unit.yr
            n_vr[smpl] = radial_velocity * unit.km / unit.s

        if print_out:
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

            tr = np.concatenate((timerange[:0:-1] * (-1), timerange))

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

            tr = timerange[::-1]
            tr[:-1] *= (-1)

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

            tr = timerange

        if print_out:
            print(f'... returning integrated orbits as array with shape\n'
                  f' (parameters x timesteps x resamples) to class SampledCluster.'
                  f'\n   -> here: (7x{array_length}x{number_of_samples})\n')

        t = np.reshape(tr, (1, len(tr))).repeat(number_of_samples, 0)
        # setting columns in array with orbit calculations
        # (x is already corrected for different standard direction)
        pt_array[0, :, :] = np.transpose(t)
        pt_array[1, :, :] = np.transpose(x)
        pt_array[2, :, :] = np.transpose(y)
        pt_array[3, :, :] = np.transpose(z)
        pt_array[4, :, :] = np.transpose(u)
        pt_array[5, :, :] = np.transpose(v)
        pt_array[6, :, :] = np.transpose(w)

        return SampledCluster(pt_array)


class SampledCluster:
    """Storing sampled integrated orbits.

    SampledCluster stores the results of N sampled orbits (:meth:`Cluster.sample_orbits`) as an array.
    Additionally, it provides several methods to summarise the results and store them in a :class:`pandas.DataFrame`.
    This summary DataFrame can also be converted to and returned as an :class:`astropy.table.Table` or QTable,
    as well as saved as a 'csv' or 'fits' file.

    :param sampled_orbit_array: sampled traceback orbits resulting from :meth:`Cluster.sample_orbits`.
    :type sampled_orbit_array: numpy.ndarray
    """

    def __init__(self, sampled_orbit_array):
        """Constructor method
        """
        time_array = sampled_orbit_array[0, :, 0]
        self.summary_dataframe = pd.DataFrame(time_array, columns=['t'])
        self.__data = sampled_orbit_array

    def get_data(self):
        """Get array with sampled orbits

        :return: Returns sampled traceback results from ``Cluster.sample_orbits`` as an array.
        :rtype: np.ndarray

        Examples
        ========

        .. code-block::

            >>> from astropy.table import Table
            >>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
            >>> cluster = Cluster(cluster_data).sample_orbit(5, 1)

            >>> cluster_sampled_data = cluster.get_data()
            >>> type(cluster_sampled_data)
            <class 'numpy.ndarray'>
            >>> np.shape(cluster_sampled_data)
            (7, 11, 1000)
        """
        return self.__data.copy()

    def add_mean(self):
        """Add mean of sampled orbits to summary DataFrame

        Examples
        ========

        .. code-block::

            >>> from astropy.table import Table
            >>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
            >>> cluster = Cluster(cluster_data).sample_orbit(5, 1)

            >>> cluster.add_mean()
            >>> cluster.summary_dataframe.columns
            Index(['t', 'X_mean', 'Y_mean', 'Z_mean', 'U_mean', 'V_mean', 'W_mean'], dtype='object')
            >>> cluster.summary_dataframe.loc[:5, 't']
            0   -5.0
            1   -4.0
            2   -3.0
            3   -2.0
            4   -1.0
            5    0.0
            Name: t, dtype: float64
        """
        self.summary_dataframe[['X_mean', 'Y_mean', 'Z_mean', 'U_mean', 'V_mean', 'W_mean']] = np.transpose(
            np.nanmean(self.__data[1:, :, :], axis=2))

    def add_median(self):
        """Add median of sampled orbits to summary DataFrame
        """
        self.summary_dataframe[['X_median', 'Y_median', 'Z_median', 'U_median', 'V_median', 'W_median']] = np.transpose(
            np.nanmedian(self.__data[1:, :, :], axis=2))

    def add_mad(self):
        """Add median absolut deviation of sampled orbits to summary DataFrame

        Examples
        ========

        .. code-block::

            >>> from astropy.table import Table
            >>> cluster_data = Table.read('./example_data/ExampleCluster_1.fits')
            >>> cluster = Cluster(cluster_data).sample_orbit(5, 1)

            >>> cluster.add_median()
            >>> cluster.summary_dataframe.columns
            Index(['t', 'X_median', 'Y_median', 'Z_median', 'U_median', 'V_median',
                   'W_median'],
                  dtype='object')
            >>> cluster.add_mad()
            >>> cluster.summary_dataframe.columns
            Index(['t', 'X_median', 'Y_median', 'Z_median', 'U_median', 'V_median',
                   'W_median', 'X_mad', 'Y_mad', 'Z_mad', 'U_mad', 'V_mad', 'W_mad'],
                  dtype='object')
        """
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(
            self.__data[1:, :, :], np.nanmean(self.__data[1:, :, :], axis=2)[:, :, None])), axis=2)
        self.summary_dataframe[['X_mad', 'Y_mad', 'Z_mad', 'U_mad', 'V_mad', 'W_mad']] = np.transpose(
            median_absolut_deviation)

    def add_std(self):
        """Add standard deviation of sampled orbits to summary DataFrame
        """
        self.summary_dataframe[['X_std', 'Y_std', 'Z_std', 'U_std', 'V_std', 'W_std']] = np.transpose(
            np.nanstd(self.__data[1:, :, :], axis=2))

    def add_percentile(self, percentile):
        """Add percentiles of sampled orbits to summary DataFrame.

        Computes the percentile of the sampled orbits along
        the second dimension. Saves the values for each parameter per timestep with the
        percentile-percentage in the column name. If there are several percentages given, each is computed and
        returned for all parameters.
        E.g. (prctl1, prctl2) -> adds 12 column to the DataFrame (6 for prctl1 + 6 for prctl2).

        :param percentile: percentage(s) to compute percentile of.
            Needs to be between 0 and 100 (for details see numpy.percentile.) Single float or sequence of float.
        :type percentile: array-like
        """
        orbit_percentiles = np.nanpercentile(self.__data[1:, :, :], percentile, axis=2)

        if hasattr(percentile, '__len__'):
            for j, p_j in enumerate(percentile):
                self.summary_dataframe[
                    [f'X_p{p_j}', f'Y_p{p_j}', f'Z_p{p_j}',
                     f'U_p{p_j}', f'V_p{p_j}', f'W_p{p_j}']] = np.transpose(orbit_percentiles[j, :, :])

        else:
            self.summary_dataframe[
                [f'X_p{percentile}', f'Y_p{percentile}', f'Z_p{percentile}',
                 f'U_p{percentile}', f'V_p{percentile}', f'W_p{percentile}']] = np.transpose(orbit_percentiles)

    def to_astropy_table(self, include_units=False):
        """Convert DataFrame to astropy.table.Table or QTable.

        Returns an astropy.table.Table or optional as astropy.table.QTable that includes units for each column.
        [t] is assumed to be Myr, coordinates are given in pc and velocities in km/s.

        :param include_units: Default is False. If True, DataFrame is converted to a QTable that can store
            Quantities with unit information.
        :type include_units: bool, optional

        :return: astropy table, optionally with units
        :rtype: astropy.table.Table or astropy.table.QTable
        """
        astropy_table = Table.from_pandas(self.summary_dataframe)
        if include_units:
            astropy_qtable = QTable(astropy_table)
            position_columns, velocity_columns = [], []
            for col in self.summary_dataframe.columns:
                if 'X' in col.upper() or 'Y' in col.upper() or 'Z' in col.upper():
                    position_columns.append(col)
                if 'U' in col.upper() or 'V' in col.upper() or 'W' in col.upper():
                    velocity_columns.append(col)
            astropy_qtable['t'].unit = unit.Myr
            astropy_qtable[position_columns].unit = unit.pc
            astropy_qtable[velocity_columns].unit = unit.km / unit.s

            return astropy_qtable

        else:
            return astropy_table

    def save_table(self, path_to_file, file_type='csv'):
        """Save DataFrame to "csv" or "fits" file. Existing files will be overwritten.

        :param path_to_file: filename including the path to file location
        :type path_to_file: str
        :param file_type: 'csv' (default) or 'fits' file. If 'fits' is chosen,
            DataFrame is first converted to astropy.table.Table in order to then write the data to fits file.
        :type file_type: str, optional

        :raises ValueError: If file_type not 'csv' or 'fits'.
        """
        if file_type == 'fits':
            if path_to_file[-5:] != '.fits':
                path_to_file = path_to_file + '.fits'
            astropy_table = Table.from_pandas(self.summary_dataframe)
            astropy_table.write(path_to_file, format='fits', overwrite=True)

        elif file_type == 'csv':
            self.summary_dataframe.to_csv(path_to_file, index=False)

        else:
            raise ValueError(f'file_type={file_type} is not valid. Set it to be either "csv" (default) or "fits".')


class Stars:
    """Samples and integrates orbits from the input data.

    :param input_data: input file for stellar cluster members that is read by the ``read_table_to_df()`` function and
        converted to a :class:`pandas.DataFrame`. If data is provided as a numpy.ndarray dimensions must be
        (12xN) or (Nx12) with the 12 parameters being ['ra', 'ra_error', 'dec', 'dec_error', 'distance',
        'distance_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error']
        in exactly this order.
    :type input_data: str, pandas.DataFrame, astropy.table.Table, numpy.ndarray
    """
    def __init__(self, input_data):
        """Constructor method
        """
        self.data = read_table_to_df(input_data)

    def sample_orbit(self, time_end, time_step, number_of_samples=1000, direction='both',
                     reference_orbit_lsr=True, reference_object_pv=None, potential=MWPotential2014,
                     print_out=False):
        """Samples orbit integration for individual stars.

        Draws each coordinate and velocity from normal
        distributions based on their measurement as mean and the measurement uncertainty as standard deviation,
        respectively, N-times. Per star, each of the N samples are integrated and collected in a 4-dimensional
        :class:`numpy.ndarray`. The shape of the returned array is
        (number of stars x 7 (parameters) x number of timesteps x number of samples).
        The seven returned parameters are t, X, Y, Z, U, V, W.

        :param time_end: absolut of integration time given as astropy.units.Quantity with time unit.
            If not a units.Quantity it is assumed to be in Myr.
        :type time_end: astropy.units.Quantity, int
        :param time_step: size of timestep as am astropy.units.Quantity. If not a units.Quantity the value
            is assumed to be in Myr.
        :type time_step: astropy.units.Quantity, float, int
        :param number_of_samples: Number of times to sample from the normal distribution. Defaults to 1000.
        :type number_of_samples: int, float, optional
        :param direction: 'direction' of integration. Integration 'backward', 'forward' or 'both' (default).
        :type direction: str, optional
        :param reference_orbit_lsr: Defaults to True. False if LSR is not the reference frame.
            If False, it is necessary to provide position and velocities of the reference frame in the attribute
            'reference_object_pv'.
        :type reference_orbit_lsr: bool, optional
        :param reference_object_pv: Positions and velocities of the reference frame. Defaults to None.
        :type reference_object_pv: 1D array or list, optional
        :param potential: potential in which to integrate the orbit. Choose from galpy.potential and import or define
            a personalised potential
        :type potential: galpy.potential, optional
        :param print_out: If True, prints information and updates on integration. Defaults to False.
        :type print_out: bool, optional

        :return: Returns a 4-dimensional array with sampled and integrated orbits for each timestep and star
        :rtype: SampledStars

        :raises ValueError: If direction is not among "backward", "forward", or "both".
        :raises ValueError: If either of time_end or time_step is 0 (zero) or time_step is
            greater than or equal to time_end.

        |

        Examples
        ========

        Initialising all stars, sampling the position and velocities, integrating and storing in a numpy.ndarray (4D).
        Can be accessed with the ``.get_data()`` method on the result from ``sample_orbit()``.

        .. code-block::

            >>> from astropy.table import Table
            >>> import numpy as np

            >>> star_data = Table.read('./example_data/ExampleCluster_1.fits')
            >>> time_end, time_step, number_of_samples = 10, 1, 100
            >>> star_orbits = Stars(star_data).sample_orbit(time_end, time_step, number_of_samples, direction='both')
            >>> np.shape(star_orbits.get_data())
            (50, 7, 21, 100)
        """

        # convert data to pandas dataframe
        data = self.data
        number_of_stars = len(data.index)

        time_end = np.abs(time_end)
        time_step = np.abs(time_step)

        if time_end == 0:
            raise ValueError('time_end cannot be 0.')
        elif time_step == 0:
            raise ValueError('time_step cannot be 0.')
        elif time_step >= time_end:
            raise ValueError('time_step cannot be greater than or equal time_end.')

        if print_out:
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
            array_length = len(timerange) * 2 - 1
        elif (direction == 'backward') | (direction == 'forward'):
            array_length = len(timerange)
        else:
            raise ValueError(f'direction={direction} is not valid.'
                             f'Set it to one of the following: "backward", "forward", "both" (default).')

        if print_out:
            print('... using Monte Carlo-type sampling for star tracebacks with method "stellar"')
        # setting output array with diemsions
        # dim 0: number of stars
        # dim 1: 7 (for t, x, y, z, u, v, w)
        # dim 2: number of timesteps
        # dim 3: number of bootstrapping repetitions
        number_of_samples = int(number_of_samples)
        pt_array = np.zeros((number_of_stars, 7, array_length, number_of_samples))

        if print_out:
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

                t = timerange[::-1]
                t[:-1] *= (-1)

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

        if print_out:
            print(f'... returning integrated stellar orbits as array with shape\n'
                  f' (number of stars x parameters x timesteps x samples).'
                  f'\n   -> here: ({number_of_stars} x 7 x {array_length} x {number_of_samples})\n')

        return SampledStars(pt_array)


class SampledStars:
    """Storing sampled integrated orbits.

    SampledStars stores the results of N sampled orbits (:meth:`Stars.sample_orbits`) as an array.
    The shape of the stored array is (number of stars x 7 (parameters) x number of timesteps x number of samples).
    Index for the parameters is 0: t, 1: X, 2: Y, 3: Z, 4: U, 5: V, 6: W.
    Additionally, it provides several methods to summarise the results per star and store them in a
    :class:`numpy.ndarray`. The methods collapse the integrated orbit array along axis 3.

    :param sampled_orbit_array: sampled traceback orbits resulting from :meth:`Stars.sample_orbits`.
    :type sampled_orbit_array: numpy.ndarray

    |

    Examples
    ========

    To access the e.g. median values of X (index 1) for each star and all timesteps use

    .. code-block::

        >>> stars = Stars('./example_data/ExampleCluster_1.fits')
        >>> time_end, time_step, number_of_samples = 10, 1, 100
        >>> star_orbits = stars.sample_orbit( time_end, time_step, number_of_samples, direction='both')
        >>> sampled_stars = star_orbits.calculate_median()

        >>> for star in range(len(stars.data.index)):
        ...     single_star_median = sampled_stars[star, 1, :]
        ...     if star == 0:
        ...         print(f'median x positions (index 1) for star {star} per timestep ({len(single_star_median)}):')
        ...         print(np.round(single_star_median, 1))
        median x positions (index 1) for star 0 per timestep (21):
        [ 92.2  97.6 103.  108.5 114.1 119.7 125.6 131.6 137.8 144.1 150.6 157.1
         163.8 170.8 177.9 185.1 192.6 200.3 208.4 216.5 224.8]
    """

    def __init__(self, sampled_orbit_array):

        self.__data = sampled_orbit_array
        self.mean_array = None
        self.std_array = None
        self.median_array = None
        self.mad_array = None
        self.percentile_array = None

    def get_data(self):
        """Get array with sampled orbits

        :return: Returns sampled traceback results from ``Cluster.sample_orbits`` as an array.
        :rtype: np.ndarray
        """
        return self.__data.copy()

    def calculate_mean(self):
        """Calculates the mean of sampled orbits per star

        :return: Returns array with mean values for each star and timestep that are stored in axis 2.
        :type: numpy.ndarray
        """
        self.mean_array = np.nanmean(self.__data, axis=3)
        return self.mean_array

    def calculate_median(self):
        """Calculates the median of sampled orbits per star

        :return: Returns array with median values for each star and timestep that are stored in axis 2.
        :type: numpy.ndarray
        """
        self.median_array = np.nanmedian(self.__data, axis=3)
        return self.median_array

    def calculate_mad(self):
        """Calculates the median absolut deviation of sampled orbits per star

        :return: Returns array with median absolut deviation values for each star and timestep that are stored
            in axis 2.
        :type: numpy.ndarray
        """
        median_absolut_deviation = np.nanmedian(np.abs(np.subtract(
            self.__data, np.nanmean(self.__data, axis=3)[:, :, :, None])), axis=3)
        self.mad_array = median_absolut_deviation
        return self.mad_array

    def calculate_std(self):
        """Calculates the standard deviation of sampled orbits per star

        :return: Returns array with standard deviation values for each star and timestep that are stored in axis 2.
        :type: numpy.ndarray
        """
        self.std_array = np.nanstd(self.__data, axis=3)
        return self.std_array

    def calculate_percentile(self, percentile):
        """Calculates the percentiles of the given percentages for the sampled orbits per star.

        If there are several percentages given, each is computed and returned for all parameters as a 4-dimensional
        array. E.g. (prctl1, prctl2) -> shape = (number of percentiles, number of stars, 7, number of timesteps).
        Access each percentile per star per timestep by indexing the axis 0 (``percentile_array[x, :, :, :]``).

        :param percentile: percentage(s) to compute percentile of.
            Needs to be between 0 and 100 (for details see numpy.percentile.) Single float or sequence of float.
        :type percentile: array-like

        :return: 3-dimensional array if percentile is single value. 4-dimensional array
            if percentile is a list of values.
            Shape is (number of percentiles, number of stars, 7, number of timesteps).
        :type: numpy.ndarray
        """
        self.percentile_array = np.nanpercentile(self.__data, percentile, axis=3)
        return self.percentile_array


def read_table_to_df(filepath_or_table):
    """Reading in a table and/or converting the input to a pandas.DataFrame, if it is not already.

    If there is no column named "distance" in the input table, and there is a column named "parallax", the distance is
    calculated with :class:àstropy.coordinates.Distance`. It samples from a normal distribution based on the parallax
    as mean and the parallax_error as standard deviation for 1000 samples. Then each is converted to a distance and
    finally, the mean and standard deviation from the 1000 samples are used as distance and distance_error
    for each star.

    :param filepath_or_table: path to saved file, or table/ array. All versions need headers including
        ['ra', 'ra_error', 'dec', 'dec_error', 'distance'/ 'parallax', 'distance_error'/ 'parallax_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error'].
        If input is an array, this order must be kept for the 12 columns/ rows.
    :type filepath_or_table: str, np.ndarray, pd.DataFrame, astropy.table.Table

    :return: DataFrame of input
    :rtype: pd.DataFrame

    :raises TypeError: If input is neither a str, pandas.DataFrame, astropy.table.Table, or numpy.ndarray
    :raises ValueError: If input table has no column 'distance' or 'parallax'.
    """
    column_names = ['ra', 'ra_error', 'dec', 'dec_error', 'distance', 'distance_error',
                    'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error']
    distance_sample = 1000
    # if input is a str it is assumed to be a path to the file location
    # file is loaded as Table, making it independent of file format
    if isinstance(filepath_or_table, str):
        itable = Table.read(filepath_or_table)
        if 'distance' not in itable.colnames:
            if 'parallax' in itable.colnames:
                print('No column named "distance" found.'
                      'Sampling "distance" and "distance_error" from "parallax" and "parallax_error".')
                n_plx = np.random.normal(itable['parallax'], itable['parallax_error'],
                                         (distance_sample, len(itable['parallax'])))
                d = Distance(parallax=n_plx * unit.mas)
                itable['distance'] = np.nanmean(d.value, axis=0)
                itable['distance_error'] = np.nanstd(d.value, axis=0)
            else:
                raise ValueError('No column named "parallax" or "distance" in input table.')
        df = itable.to_pandas()

    # if input is a astropy.table.Table, it is converted to dataframe
    elif isinstance(filepath_or_table, Table):
        if 'distance' not in filepath_or_table.colnames:
            if 'parallax' in filepath_or_table.colnames:
                print('No column named "distance" found.'
                      'Sampling "distance" and "distance_error" from "parallax" and "parallax_error".')
                n_plx = np.random.normal(filepath_or_table['parallax'], filepath_or_table['parallax_error'],
                                         (distance_sample, len(filepath_or_table['parallax'])))
                d = Distance(parallax=n_plx * unit.mas)
                filepath_or_table['distance'] = np.nanmean(d.value, axis=0)
                filepath_or_table['distance_error'] = np.nanstd(d.value, axis=0)
            else:
                raise ValueError('No column named "parallax" or "distance" in input table.')
        df = filepath_or_table.to_pandas()

    # if input is a dataframe, it will be returned as such
    elif isinstance(filepath_or_table, pd.DataFrame):
        if 'distance' not in filepath_or_table.columns:
            if 'parallax' in filepath_or_table.colnames:
                print('No column named "distance" found.'
                      'Sampling "distance" and "distance_error" from "parallax" and "parallax_error".')
                n_plx = np.random.normal(filepath_or_table['parallax'], filepath_or_table['parallax_error'],
                                         (distance_sample, len(filepath_or_table['parallax'])))
                d = Distance(parallax=n_plx * unit.mas)
                filepath_or_table['distance'] = np.nanmean(d.value, axis=0)
                filepath_or_table['distance_error'] = np.nanstd(d.value, axis=0)
            else:
                raise ValueError('No column named "parallax" or "distance" in input table.')
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
        raise TypeError(f'{np.dtype(filepath_or_table)} is not valid. Data input must be either a path to a table, '
                        f'a pandas DataFrame, an astropy.table.Table, '
                        f'or a 2D array with shape being either (Nx12) or (12xN).')

    return df


def traceback_stars_radec(sky_object_pv, t, reference_orbit_lsr=True, reference_object_pv=None, back=True,
                          potential=MWPotential2014):
    """Integrates coordinates and velocities relative to a reference frame.

    Calculate the traceback of one or more "sky objects" initialised by heliocentric equatorial coordinates
    (as published by *GAIA*). Output is in Cartesian coordinates with the center of the coordinate system being the
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
    :type reference_orbit_lsr: bool, optional
    :param reference_object_pv: position and velocity of reference object if reference_orbit is *False*
    :type reference_object_pv: list, optional
    :param back: default *True*, integrates backward in time (flips time sequence and velocities).
        If back is *False* integrates forward in time.
    :type back: bool, optional
    :param potential: default MWPotential2014, any other galpy potential can be passed
        (https://docs.galpy.org/en/latest/potential.html)
    :type potential: galpy.potential, optional

    :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s.
     For each coordinate an array is returned with shape (len(t), len(sky_object_pv[any]):
     for each timestep and sky object integrated positions and velocities are returned.
    :rtype: float, array

    :raises ValueError: If "reference_orbit_lsr" is set to False but no values passed to "reference_object_pv".
    :raises ValueError: If "reference_orbit_lsr" neither True nor False.
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
        raise ValueError('"reference_orbit_lsr" is set to False but no reference frame is provided in the '
                         'parameter "reference_object_pv".')

    elif (reference_object_pv is not None) & (reference_orbit_lsr is True):
        del reference_object_pv

    else:
        raise ValueError('Reference orbit not defined. Set either "reference_orbit_lsr" to True or set it to'
                         'False and provide coordinates for a reference frame in "reference_object_pv".')

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


def skycoord_from_table(path_to_file):
    """Creates a SkyCoord object from table data.

    Function to create a 6D astropy.coordinates.SkyCoord from the input table
    (using ra, dec, parallax/ distance, pmra, pmdec, radial velocity). If no column is called 'distance',
    parallax is automatically converted to distance by Monte Carlo-type sampling from the measurement and measurement
    uncertainty of the parallax.

    :param path_to_file: path to table file
    :type path_to_file: str

    :return: 6D SkyCoord object based on the data in the table
    :rtype: astropy.coordinates.SkyCoord

    :raises: KeyError: If the table has no column named 'distance' or 'parallax'.
    """
    itable = Table.read(path_to_file)
    column_names = itable.colnames

    if 'distance' in column_names:
        dist = itable['distance'].value * unit.kpc
    elif ('distance' not in column_names) & ('parallax' in column_names):
        dist = Distance(itable['parallax']).to_value(unit.kpc)
    else:
        raise KeyError('Table has no column named "distance" or "parallax".')

    skycoord_object = SkyCoord(ra=unit.Quantity(itable['ra'].value * unit.deg, copy=False),
                               dec=unit.Quantity(itable['dec'].value * unit.deg, copy=False),
                               distance=unit.Quantity(dist, unit.kpc, copy=False),
                               pm_ra_cosdec=unit.Quantity(itable['pmra'].value * unit.mas / unit.yr, copy=False),
                               pm_dec=unit.Quantity(itable['pmdec'].value * unit.mas / unit.yr, copy=False),
                               radial_velocity=unit.Quantity(itable['radial_velocity'].value * unit.km / unit.s,
                                                             copy=False))

    return skycoord_object
