load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"
load "/usr/share/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
  
  if (res.eq."f19") then
    n_lat = 96
    n_lon = 144
  end if  
  if (res.eq."g16") then
    n_lat = 384
    n_lon = 320
  end if
  if (res.eq."1x1d") then
    n_lat = 180
    n_lon = 360
  end if

  srcGridFile = "/data/bnb32/inputdata/grid_files/src_grid_tmp.nc"
  dstGridFile = "/data/bnb32/inputdata/grid_files/ds_grid_tmp.nc"
  wgtFile = "/data/bnb32/inputdata/mapping_files/map_grids_tmp.nc"
  
  fin = addfile(infile, "rw")
  fout = addfile(outfile, "cw")
  
  dimNames = (/"lat", "lon"/)
  dimSizes = (/n_lat, n_lon/)
  dimUnlim = (/False, False/)
  filedimdef(fout, dimNames, dimSizes, dimUnlim)
  
  landmask = fin->landmask
  oceanmask = fin->oceanmask
  height = fin->height
  depth = fin->depth
  PHIS = fin->PHIS
  
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new(dimsizes(fin->landmask),integer,1)
  
  lat  = latGlobeF(1801, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(3601, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(srcGridFile,lat,lon,opt)
  
  delete(opt)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@Check = True
  
  opt@GridMask = new((/n_lat,n_lon/),integer,1)
      
  lat  = latGlobeFo(n_lat, "lat", "latitude", "degrees_north")
  lon  = lonGlobeFo(n_lon, "lon", "longitude", "degrees_east")
  
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

  remap_lndmask = ESMF_regrid_with_weights(landmask,wgtFile,opt)
  remap_ocnmask = ESMF_regrid_with_weights(oceanmask,wgtFile,opt)
  remap_depth = ESMF_regrid_with_weights(depth,wgtFile,opt)
  remap_height = ESMF_regrid_with_weights(height,wgtFile,opt)
  remap_PHIS = ESMF_regrid_with_weights(PHIS,wgtFile,opt)
   
  ;replace_ieeenan(remap_frac,0.0,0)
  ;remap_mask = where(remap_frac>0.9,1,0)

  fout->oceanmask = (/remap_ocnmask/)
  fout->landmask = (/remap_lndmask/)
  fout->depth = (/remap_depth/)
  fout->height = (/remap_height/)
  fout->PHIS = (/remap_PHIS/)

end
