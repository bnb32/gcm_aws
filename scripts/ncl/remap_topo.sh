#!/bin/bash
export NCL_POP_REMAP='/data/bnb32/inputdata/mapping_files'

ncl infile='"../tmp_in.nc"' outfile='"../tmp_out.nc"' ./remap_topo.ncl
