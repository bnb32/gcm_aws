"""Regridder module"""


def regridder_module(land, ds_out):
    """xesmf regridder"""
    import xesmf as xe
    return xe.Regridder(land, ds_out, 'bilinear', ignore_degenerate=True)
