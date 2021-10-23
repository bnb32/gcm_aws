load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"
load "/usr/share/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
  
  ;srcGrdFile = "/data/bnb32/inputdata/grid_files/fv1.9x2.5_141008.nc"
  srcGridFile = "/data/bnb32/inputdata/grid_files/fv1.9x2.5_tmp.nc"
  dstGridFile = "/data/bnb32/inputdata/grid_files/gx1v6_tmp.nc"
  ;dstGridFile = "/data/bnb32/inputdata/grid_files/gx1v6_090205.nc"
  wgtFile = "/data/bnb32/inputdata/mapping_files/map_fv1.9x2.5_to_gx1v6_tmp.nc"
  
  fin = addfile(infile, "rw")
  fout = addfile(outfile, "w")
  
  landfrac = fin->frac
  landmask = fin->mask
  oceandepth = fin->depth
  
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new(dimsizes(fin->mask),integer,1)
  
  lat  = latGlobeF(96, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(144, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(srcGridFile,lat,lon,opt)
  
  delete(opt)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new(dimsizes(fout->mask),integer,1)
  opt@Check = True
  
  lat  = latGlobeFo(384, "lat", "latitude", "degrees_north")
  lon  = lonGlobeFo(320, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(dstGridFile,lat,lon,opt)
  
  delete(opt)
  opt = True
  opt@ForceOverwrite = True
  opt@InterpMethod = "bilinear"
  opt@IgnoreUnmappedPoints = False
  opt@SrcRegional = False
  opt@DstRegional = False
  opt@Check = True

  ESMF_regrid_gen_weights(srcGridFile,dstGridFile,wgtFile,opt)

  delete(opt)
  opt = True

  remap_frac = ESMF_regrid_with_weights(1-landfrac,wgtFile,opt)
  remap_depth = ESMF_regrid_with_weights(oceandepth,wgtFile,opt)
   
  printVarSummary(remap_frac)

  replace_ieeenan(remap_frac,0.0,0)
  remap_mask = where(remap_frac>0.9,1,0)

  fout->mask = (/remap_mask/)
  fout->frac = (/remap_frac/)
  fout->depth = (/remap_depth/)
  
end
