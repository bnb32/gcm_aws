"""Ecrlgcm data module"""
import os

from ecrlgcm import DATA_DIR

def get_co2_series():
    f = open(os.path.join(DATA_DIR,'co2.txt'),'r')
    lines = f.readlines()
    lines = [l.strip('\n').split() for l in lines]
    lines = {-float(l[0]): float(l[1])*300.0 for l in lines}
    return lines

def get_obl_series():
    f = open(os.path.join(DATA_DIR,'orbit.txt'),'r')
    lines = f.readlines()[3:]
    lines = [l.strip('\n').split() for l in lines]
    lines = {-float(l[0])/1000: float(l[3]) for l in lines}
    return lines

def get_ecc_series():
    f = open(os.path.join(DATA_DIR,'orbit.txt'),'r')
    lines = f.readlines()[3:]
    lines = [l.strip('\n').split() for l in lines]
    lines = {-float(l[0])/1000: float(l[1]) for l in lines}
    return lines

co2_series = get_co2_series()
ecc_series = get_ecc_series()
obl_series = get_obl_series()
