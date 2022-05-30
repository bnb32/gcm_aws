"""Install needed packages"""
import os

cmd = 'sudo yum-config-manager --enable epel; '
cmd += 'sudo yum install '
cmd += 'netcdf '
cmd += 'netcdf-fortran '

os.system(cmd)

cmd = 'conda install -n gcm_env -c conda-forge '
cmd += 'xarray '
cmd += 'xesmf '
cmd += 'netCDF4 '

os.system(cmd)
