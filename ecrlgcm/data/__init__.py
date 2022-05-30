"""Ecrlgcm data module"""
import os

from ecrlgcm import DATA_DIR


def get_co2_series():
    """Get CO2 time series"""
    with open(os.path.join(DATA_DIR, 'co2.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split() for line in lines]
        lines = {-float(line[0]): float(line[1]) * 300.0 for line in lines}
    return lines


def get_obl_series():
    """Get orbital obliquity time series"""
    with open(os.path.join(DATA_DIR, 'orbit.txt'), 'r') as f:
        lines = f.readlines()[3:]
        lines = [line.strip('\n').split() for line in lines]
        lines = {-float(line[0]) / 1000: float(line[3]) for line in lines}
    return lines


def get_ecc_series():
    """Get orbital eccentricity time series"""
    with open(os.path.join(DATA_DIR, 'orbit.txt'), 'r') as f:
        lines = f.readlines()[3:]
        lines = [line.strip('\n').split() for line in lines]
        lines = {-float(line[0]) / 1000: float(line[1]) for line in lines}
    return lines


co2_series = get_co2_series()
ecc_series = get_ecc_series()
obl_series = get_obl_series()
