"""Cartopy utils"""


def plate_carree(**kwargs):
    """plate carree from cartopy"""
    import cartopy.crs as ccrs
    return ccrs.PlateCarree(kwargs)


def orthographic(**kwargs):
    """orthographic from cartopy"""
    import cartopy.crs as ccrs
    return ccrs.Orthographic(kwargs)


def get_projection(globe, **kwargs):
    """get cartopy projection"""
    import cartopy.crs as ccrs
    if globe:
        proj = ccrs.Orthographic(kwargs)
    else:
        proj = ccrs.PlateCarree(kwargs)
    return proj
