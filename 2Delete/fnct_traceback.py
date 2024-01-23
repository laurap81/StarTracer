from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from astropy.coordinates import Galactocentric, LSR
import astropy.units as unit
import numpy as np


def traceback_stars_radec(ra, dec, distance, pm_ra_cosdec, pmdec, vr, t, reference_orbit='sun', back=True):
    """
    code to calculate the traceback of one or more stars. It needs (single/ an array/ list of):

    :param ra: right ascension
    :param dec: declination
    :param distance: distance
    :param pm_ra_cosdec: proper motion in ra
    :param pmdec: proper motion in declination
    :param vr: radial velocity
    :param t: array, time values at which to evaluate the orbit (0 to X)
    :param reference_orbit: default 'sun', else use a list or array with the reference frames parameters like [ra, dec, distance, pmra, pmdec, vr]
    :param back: bool, backwards integration (flips time array), default is True
    :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s
    """
    gc = Galactocentric()

    # orbit of stars
    os = Orbit([ra, dec, distance.to(unit.kpc), pm_ra_cosdec, pmdec, vr],
               radec=True,
               uvw=False,
               ro=gc.galcen_distance,
               zo=gc.z_sun,
               vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
               solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    # observer's orbit (LSR)
    if reference_orbit == 'sun':
        ob = Orbit([0 * unit.deg, 0 * unit.deg, 0 * unit.kpc, -LSR.v_bary.d_x, -LSR.v_bary.d_y, -LSR.v_bary.d_z],
                   radec=True,
                   uvw=True,
                   ro=gc.galcen_distance,
                   zo=0,
                   vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                   solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))
    else:
        ob = Orbit([reference_orbit[0], reference_orbit[1], reference_orbit[2].to(unit.kpc),
                    reference_orbit[3], reference_orbit[4], reference_orbit[5]],
                   radec=True,
                   uvw=False,
                   ro=gc.galcen_distance,
                   zo=0,
                   vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                   solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    # if using 'back=True' function will return backwards integration
    # if using 'back=False' function will integrate into the future
    if back:
        os.flip(inplace=True)
        ob.flip(inplace=True)

    # integration with chosen potential
    os.integrate(t=t, pot=MWPotential2014)
    ob.integrate(t=t, pot=MWPotential2014)

    # setting output arrays as 'orbit of stars - orbit of observer'
    # at the given times
    x = os.x(t) - ob.x(t)
    y = os.y(t) - ob.y(t)
    z = os.z(t) - ob.z(t)

    if back:
        u = os.vx(t) - ob.vx(t)
        v = - (os.vy(t) - ob.vy(t))
        w = - (os.vz(t) - ob.vz(t))
    else:
        u = - (os.vx(t) - ob.vx(t))
        v = os.vy(t) - ob.vy(t)
        w = os.vz(t) - ob.vz(t)
    #
    # elif ra.size > 1:
    #     x = os.x(t) - ob.x(t)
    #     y = os.y(t) - ob.y(t)
    #     z = os.z(t) - ob.z(t)
    #
    #     if back:
    #         u = os.vx(t) - ob.vx(t)
    #         v = - (os.vy(t) - ob.vy(t))
    #         w = - (os.vz(t) - ob.vz(t))
    #     else:
    #         u = - (os.vx(t) - ob.vx(t))
    #         v = os.vy(t) - ob.vy(t)
    #         w = os.vz(t) - ob.vz(t)
    # else:
    #     print('Input is neither a float, list or array. (ra, dec, ... )')
    #     x, y, z, u, v, w, = [], [], [], [], [], []

    # return coordinates as array in pc
    return x * (-1e3), y * 1e3, z * 1e3, u, v, w


def traceback_stars_galactic(glon, glat, distance, pm_ll, pm_bb, vr, t, reference_orbit='sun',
                             reference_frame='galactic', back=True):
    """
    code to calculate the traceback of one or more stars. It needs (single/ an array/ list of):

    :param glon: galactic longitude
    :param glat: galactic latitude
    :param distance: distance
    :param pm_ll: proper motion l
    :param pm_bb: proper motion b
    :param vr: radial velocity
    :param t: array, time values at which to evaluate the orbit (0 to X)
    :param reference_orbit: default 'sun', else use a list or array with the reference frames parameters like [glon, glat, distance, pmll, pmbb, vr]
    :param reference_frame: default 'galactic', others are 'radec'
    :param back: bool, backwards integration (flips time array), default is True
    :return: x (increasing towards galactic center), y, z in pc and u, v, w in km/s
    """
    gc = Galactocentric()

    # orbit of stars
    os = Orbit([glon, glat, distance.to(unit.kpc), pm_ll, pm_bb, vr],
               lb=True,
               radec=False,
               uvw=False,
               ro=gc.galcen_distance,
               zo=gc.z_sun,
               vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
               solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    # observer's orbit (LSR)
    if reference_orbit == 'sun':
        ob = Orbit([0 * unit.deg, 0 * unit.deg, 0 * unit.kpc, -LSR.v_bary.d_x, -LSR.v_bary.d_y, -LSR.v_bary.d_z],
                   radec=True,
                   uvw=True,
                   ro=gc.galcen_distance,
                   zo=0,
                   vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                   solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))
    else:
        ob = Orbit([reference_orbit[0], reference_orbit[1], reference_orbit[2].to(unit.kpc),
                    reference_orbit[3], reference_orbit[4], reference_orbit[5]],
                   radec=True,
                   uvw=False,
                   ro=gc.galcen_distance,
                   zo=0,
                   vo=gc.galcen_v_sun.d_y - LSR.v_bary.d_y,
                   solarmotion=unit.Quantity([-LSR.v_bary.d_x, LSR.v_bary.d_y, LSR.v_bary.d_z]))

    # if using 'back=True' function will return backwards integration
    # if using 'back=False' function will integrate into the future
    if back:
        os.flip(inplace=True)
        ob.flip(inplace=True)

    # integration with chosen potential
    os.integrate(t=t, pot=MWPotential2014)
    ob.integrate(t=t, pot=MWPotential2014)

    # setting output arrays as 'orbit of stars - orbit of observer'
    # at the given times
    x = os.x(t) - ob.x(t)
    y = os.y(t) - ob.y(t)
    z = os.z(t) - ob.z(t)

    if back:
        u = os.vx(t) - ob.vx(t)
        v = - (os.vy(t) - ob.vy(t))
        w = - (os.vz(t) - ob.vz(t))
    else:
        u = - (os.vx(t) - ob.vx(t))
        v = os.vy(t) - ob.vy(t)
        w = os.vz(t) - ob.vz(t)

    # return coordinates as array in pc
    return x * (-1e3), y * 1e3, z * 1e3, u, v, w
