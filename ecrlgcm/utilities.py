"""Ecrlgcm utilities"""

import logging
from sys import stdout
import numpy as np
from tqdm import tqdm
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


def polar_to_cartesian(radius=1, lat=0, lon=0):
    """Map polar to cartesian coordinates"""
    x = radius * np.cos(np.pi / 180 * lat) * np.cos(np.pi / 180 * lon)
    y = radius * np.cos(np.pi / 180 * lat) * np.sin(np.pi / 180 * lon)
    z = radius * np.sin(np.pi / 180 * lat)
    return {'x': x, 'y': y, 'z': z}


def mapping_map_to_sphere(lon, lat, radius=1):
    """Map lat lon to spherical coordinates"""
    lon = np.array(lon, dtype=np.float64)
    lat = np.array(lat, dtype=np.float64)
    lon = np.pi / 180.0 * lon
    lat = np.pi / 180.0 * lat
    xs = radius * np.cos(lon) * np.cos(lat)
    ys = radius * np.sin(lon) * np.cos(lat)
    zs = radius * np.sin(lat)
    return xs, ys, zs


def interp(a, b, dt):
    """Generic linear interpolation"""
    a_tmp = a[: min((a.shape[0], b.shape[0]))]
    b_tmp = b[: a_tmp.shape[0]]
    return a_tmp * (1 - dt) + dt * b_tmp


def get_logger():
    """Get available logger"""
    return logger


isca_plevels = [4.32865344, 15.54935838, 25.36458666, 39.7283521,
                59.85607174, 86.9016501, 121.79748549, 165.08883856,
                216.79319817, 276.31361584, 342.42730307, 413.35702347,
                486.91737837, 560.71355652, 632.36253604, 699.70472315,
                760.97778738, 814.93562293, 860.90496012, 898.78288051,
                928.98741429, 952.37693492, 970.15513959, 983.77727078,
                994.87004887]

cesm_plevels = [3.64346569, 7.59481965, 14.35663225, 24.61222,
                35.92325002, 43.19375008, 51.67749897, 61.52049825,
                73.75095785, 87.82123029, 103.31712663, 121.54724076,
                142.99403876, 168.22507977, 197.9080867, 232.82861896,
                273.91081676, 322.24190235, 379.10090387, 445.9925741,
                524.68717471, 609.77869481, 691.38943031, 763.40448111,
                820.85836865, 859.53476653, 887.02024892, 912.64454694,
                936.19839847, 957.48547954, 976.32540739, 992.55609512]


def land_year_range(arg):
    """Get allowable range of years"""
    try:
        if int(arg) == float(arg):
            f = int(arg)
    except Exception:
        try:
            f = float(arg)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'land_year must be float or integer') from e

    if f < min_land_year or f > max_land_year:
        raise argparse.ArgumentTypeError(
            f'land_year must be < {max_land_year} and > {min_land_year}')
    return f


def none_or_str(arg):
    """Get none or string"""
    if arg == 'None':
        return None
    return arg


def none_or_int(arg):
    """Get none or int"""
    if arg == 'None':
        return None
    return int(arg)


def none_or_float(arg):
    """Get none or float"""
    if arg == 'None':
        return None
    return float(arg)


def sig_round(number, figs):
    """Round to specified number of sig figs"""
    return float('%s' % float(f'%.{figs}g' % number))


def sliding_std(data, dx=3, dy=3):
    """Compute sliding standard deviation"""
    tmp = np.array(data)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            i_min = max(0, i - dx // 2)
            i_max = min(tmp.shape[0], i + dx // 2)
            j_min = max(0, j - dy // 2)
            j_max = min(tmp.shape[1], j + dy // 2)
            tmp[i, j] = np.std(tmp[i_min: i_max + 1, j_min: j_max + 1])
    return tmp

def count_between(array, min_val, max_val):
    """Count number of values in array betwee min / max"""
    return ((min_val <= array) & (array <= max_val)).sum()

def cell_overlap(inlat, inlon, lat, lon, lat_dx, lon_dx, landmask):
    """Compute cell overlap for given coordinates"""
    total_count = 0
    mask_count = 0

    if lon >= 180.0:
        lon -= 360.0

    max_i = np.argmin(np.abs(inlat - (lat - lat_dx / 2)))
    min_i = np.argmin(np.abs(inlat - (lat + lat_dx / 2)))

    min_j = np.argmin(np.abs(inlon - (lon - lon_dx / 2)))
    max_j = np.argmin(np.abs(inlon - (lon + lon_dx / 2)))

    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if ((np.abs(inlat[i] - lat) < lat_dx / 2) and
                (np.abs(inlon[j] - lon) < lon_dx / 2)):
                total_count += 1
                if landmask[i, j] > 0:
                    mask_count += 1
    return mask_count / total_count

def overlap_fraction(inlat, inlon, outlat, outlon, landmask):
    """Compute overlap fractio between specified coordinate sets"""
    tmp = np.zeros((len(outlat), len(outlon)))

    lat_dx = outlat[1] - outlat[0]
    lon_dx = outlon[1] - outlon[0]

    for i in tqdm(range(len(outlat))):
        for j in range(len(outlon)):
            tmp[i, j] = cell_overlap(inlat, inlon, outlat[i], outlon[j],
                                     lat_dx, lon_dx, landmask)
    return tmp
