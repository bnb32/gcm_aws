#!/bin/bash
export NCL_POP_REMAP='/data/bnb32/inputdata/mapping_files'

#ncl infile='"/data/bnb32/inputdata/topo_files/cesm/topo_230Ma_f19_g16.nc"' outfile='"/home/ec2-user/environment/gcm_aws/ecrlgcm/data/tmp/tmp.nc"' ./remap_topo.ncl
#ncl infile='"/data/bnb32/inputdata/topo_files/cesm/topo_230Ma_f19_g16.nc"' outfile='"/home/ec2-user/environment/gcm_aws/ecrlgcm/data/tmp/oceanfrac.nc"' ./remap_topo.ncl
ncl infile='"/data/bnb32/inputdata/landfrac_files/landfrac_230Ma_f19_g16.nc"' outfile='"/home/ec2-user/environment/gcm_aws/ecrlgcm/data/tmp/oceanfrac.nc"' ./remap_topo.ncl
#ncl infile='"/data/cesm/inputdata/share/domains/domain.lnd.fv1.9x2.5_gx1v6.090206.nc"' outfile='"/home/ec2-user/environment/gcm_aws/ecrlgcm/data/tmp/oceanfrac.nc"' ./remap_topo.ncl
