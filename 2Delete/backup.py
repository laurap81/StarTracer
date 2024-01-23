# if average_method == 'mean':
#     print('... using the "mean" and "standard error" as averaging method')
#     # averaging over stellar coordinates and velocities
#     ra, ra_error = np.nanmean(sampled_data['ra']), np.nanstd(sampled_data['ra'].values) / np.sqrt(number_of_stars)
#     dec, dec_error = np.nanmean(sampled_data['dec']), np.nanstd(sampled_data['dec'].values) / np.sqrt(number_of_stars)
#     distance, distance_error = np.nanmean(sampled_data['distance']), np.nanstd(
#         sampled_data['distance'].values) / np.sqrt(number_of_stars)
#     pmra, pmra_error = np.nanmean(sampled_data['pmra']), np.nanstd(sampled_data['pmra'].values) / np.sqrt(
#         number_of_stars)
#     pmdec, pmdec_error = np.nanmean(sampled_data['pmdec']), np.nanstd(sampled_data['pmdec'].values) / np.sqrt(
#         number_of_stars)
#     radial_velocity, radial_velocity_error = np.nanmean(sampled_data['radial_velocity']), \
#         np.nanstd(sampled_data['radial_velocity'].values) / np.sqrt(number_of_stars)
#
# elif average_method == 'median':
#     print('... using the "median" and "median absolut deviation" as averaging method')
#     # averaging over stellar coordinates and velocities
#     ra, ra_error = np.nanmedian(sampled_data['ra']), mad(sampled_data['ra'].values)
#     dec, dec_error = np.nanmedian(sampled_data['dec']), mad(data['dec'].values)
#     distance, distance_error = np.nanmedian(sampled_data['distance']), mad(sampled_data['distance'].values)
#     pmra, pmra_error = np.nanmedian(sampled_data['pmra']), mad(sampled_data['pmra'].values)
#     pmdec, pmdec_error = np.nanmedian(sampled_data['pmdec']), mad(sampled_data['pmdec'].values)
#     radial_velocity, radial_velocity_error = np.nanmedian(sampled_data['radial_velocity']), \
#         mad(sampled_data['radial_velocity'].values)
#
# else:
#     ra, ra_error = 0, 0
#     dec, dec_error = 0, 0
#     distance, distance_error = 0, 0
#     pmra, pmra_error = 0, 0
#     pmdec, pmdec_error = 0, 0
#     radial_velocity, radial_velocity_error = 0, 0
#     raise 'function arg "average_method" is neither set to "mean" or "median".'


########################################################################################################################
# def traceback_stars_galactic(sky_object_pv: list, t: np.ndarray or list, reference_orbit_lsr=True,
#                              reference_object_pv=None, back=True, potential=MWPotential2014):
#     """
#     code to calculate the traceback of one or more sky_objects initialised by heliocentric galactic coordinates.
#     Output is in Cartesian coordinates with the center of the coordinate system being the reference frame given.
#
#
#     :param list[int, float, Quantity] sky_object_pv: position and velocity of traceable object(s),
#     in the form [glon, glat, distance, pmll, pmbb, radial velocity].
#     Single values or array of values for each coordinate. Can but do not have to be astropy.units.Quantity.
#     :param any t: time value(s) at which to evaluate the orbit (0 to t_n)
#     :param bool reference_orbit_lsr: default *True* for LSR as reference frame
#     (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s.
#     If *False*, [glon, glat, distance, pmll, pmbb, radial velocity] needs to be passed to the reference_object_pv
#     parameter.
#     :param list reference_object_pv: position and velocity of reference object if reference_orbit is *False*
#     :param bool back: default *True*, integrates backward in time (flips time sequence and velocities).
#     If back is *False* integrates forward in time.
#     :param galpy.potential potential: default MWPotential2014, any other galpy potential can be passed
#     (https://docs.galpy.org/en/latest/potential.html)
#
#     :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s.
#     For each coordinate an array is returned with shape (len(t), len(sky_object_pv[any]):
#     for each timestep and sky object integrated positions and velocities are returned.
#     :rtype: float, array
#     """
#
#     gc = Galactocentric()
#
#     glon_so = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#     glat_so = unit.Quantity(sky_object_pv[1], unit.deg, copy=False)
#     distance_so = unit.Quantity(sky_object_pv[2], unit.kpc, copy=False)
#     pml_so = unit.Quantity(sky_object_pv[3], unit.mas / unit.yr, copy=False)
#     pmb_so = unit.Quantity(sky_object_pv[4], unit.mas / unit.yr, copy=False)
#     radialvelocity_so = unit.Quantity(sky_object_pv[5], unit.km / unit.s, copy=False)
#
#     t = unit.Quantity(t, unit.Myr, copy=False)
#
#     # reference frame or observer's orbit
#     if reference_orbit_lsr:
#         # reference frame is the LSR at (X, Y, Z) = (0, 0, 0) pc and (U, V, W) = (-11.1, -12.24, -7.25) km/s
#         reference_orbit = Orbit(
#             [0 * unit.deg, 0 * unit.deg, 0 * unit.kpc, -LSR.v_bary.d_x, -LSR.v_bary.d_y, -LSR.v_bary.d_z],
#             radec=False,
#             lb=True,
#             uvw=True,
#             ro=gc.galcen_distance,
#             zo=0,
#             vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
#             solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))
#
#     elif (reference_object_pv is not None) & (reference_orbit_lsr is False):
#         glon_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#         glat_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#         distance_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#         pml_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#         pmb_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#         radialvelocity_ro = unit.Quantity(sky_object_pv[0], unit.deg, copy=False)
#
#         # custom reference frame
#         reference_orbit = Orbit([glon_ro, glat_ro, distance_ro, pml_ro, pmb_ro, radialvelocity_ro],
#                                 radec=False,
#                                 lb=True,
#                                 uvw=False,
#                                 ro=gc.galcen_distance,
#                                 zo=0,
#                                 vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
#                                 solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))
#
#     if (reference_object_pv is None) & (reference_orbit_lsr is False):
#         raise 'reference_orbit_lsr is set to False but no reference frame is provided in the ' \
#               'parameter reference_object_pv.'
#
#     elif (reference_object_pv is not None) & (reference_orbit_lsr is True):
#         del reference_object_pv
#
#     else:
#         raise 'Reference orbit not defined. Set either reference_orbit_lsr to True or set it to' \
#               'False and provide coordinates for a reference frame in reference_object_pv.'
#
#     # orbit of sky object(s)
#     skyobject_orbit = Orbit([glon_so, glat_so, distance_so, pml_so, pmb_so, radialvelocity_so],
#                             radec=False,
#                             lb=True,
#                             uvw=False,
#                             ro=gc.galcen_distance,
#                             zo=gc.z_sun,
#                             vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
#                             solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))
#
#     # if using 'back=True' function will return backwards integration
#     # if using 'back=False' function will integrate into the future
#     if back:
#         skyobject_orbit.flip(inplace=True)
#         reference_orbit.flip(inplace=True)
#
#     # integration with chosen potential
#     skyobject_orbit.integrate(t=t, pot=potential)
#     reference_orbit.integrate(t=t, pot=potential)
#
#     # setting output arrays as 'orbit of stars - orbit of observer'
#     # at the given times
#     x = skyobject_orbit.x(t) - reference_orbit.x(t)
#     y = skyobject_orbit.y(t) - reference_orbit.y(t)
#     z = skyobject_orbit.z(t) - reference_orbit.z(t)
#
#     if back:
#         u = skyobject_orbit.vx(t) - reference_orbit.vx(t)
#         v = - (skyobject_orbit.vy(t) - reference_orbit.vy(t))
#         w = - (skyobject_orbit.vz(t) - reference_orbit.vz(t))
#     else:
#         u = - (skyobject_orbit.vx(t) - reference_orbit.vx(t))
#         v = skyobject_orbit.vy(t) - reference_orbit.vy(t)
#         w = skyobject_orbit.vz(t) - reference_orbit.vz(t)
#
#     # return coordinates as array in pc
#     return x * (-1e3), y * 1e3, z * 1e3, u, v, w
########################################################################################################################