"""Setup script"""
from distutils.core import setup

setup(
    name='ecrlgcm',
    version='0.1.0',
    url='https://github.com/bnb32/gcm_aws',
    author='Brandon N. Benton',
    description='for setting up and running gcms on aws',
    packages=['ecrlgcm'],
    package_dir={'ecrlgcm': './ecrlgcm'},
    install_requires=[
        'matplotlib',
        'numpy',
        'xarray',
        'proj',
        'geos',
        'pop-tools',
        'netCDF4',
        'metpy',
        'tqdm',
        'sphinx-argparse',
        'joblib',
        'dash_daq'
        'f90nml',
        'isca @ git+https://github.com/ExeClim/Isca.git#egg=isca',
        'xesmf @ git+https://github.com/pangeo-data/xesmf.git']
)
