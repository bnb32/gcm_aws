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
  if (res.eq."f1") then
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
  
  landmask = lonFlip(fin->landmask)
  landfrac = lonFlip(fin->landfrac)
  oceanmask = lonFlip(fin->oceanmask)
  oceanfrac = lonFlip(fin->oceanfrac)
  height = lonFlip(fin->height)
  depth = lonFlip(fin->depth)
  PHIS = lonFlip(fin->PHIS)
  z = lonFlip(fin->z)
  
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new(dimsizes(fin->landmask),integer,1)
  
  lat  = latGlobeF(n_lat_in, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(n_lon_in, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(srcGridFile,lat,lon,opt)
  
  delete(opt)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@Check = True
  opt@GridMask = new((/n_lat,n_lon/),integer,1)
  
  if (res.eq."f19") then
    lat  = latGlobeF(96, "lat", "latitude", "degrees_north")
    lon  = lonGlobeF(144, "lon", "longitude", "degrees_east")
  end if  
  if (res.eq."g16") then
    lat  = latGlobeFo(384, "lat", "latitude", "degrees_north")
    lon  = lonGlobeFo(320, "lon", "longitude", "degrees_east")
  end if
  if (res.eq."f1") then
    lat  = latGlobeF(180, "lat", "latitude", "degrees_north")
    lon  = lonGlobeF(360, "lon", "longitude", "degrees_east")
  end if
      
  rectilinear_to_SCRIP(dstGridFile,lat,lon,opt)
  
  delete(opt)
  opt = True
  opt@ForceOverwrite = True
  opt@InterpMethod = "conserve"
  opt@IgnoreUnmappedPoints = False
  opt@SrcRegional = False
  opt@DstRegional = False
  opt@Check = True

  ESMF_regrid_gen_weights(srcGridFile,dstGridFile,wgtFile,opt)

  delete(opt)
  opt = True

  remap_lndmask = ESMF_regrid_with_weights(landmask,wgtFile,opt)
  remap_lndfrac = ESMF_regrid_with_weights(landfrac,wgtFile,opt)
  remap_ocnmask = ESMF_regrid_with_weights(oceanmask,wgtFile,opt)
  remap_ocnfrac = ESMF_regrid_with_weights(oceanfrac,wgtFile,opt)
  remap_depth = ESMF_regrid_with_weights(depth,wgtFile,opt)
  remap_height = ESMF_regrid_with_weights(height,wgtFile,opt)
  remap_PHIS = ESMF_regrid_with_weights(PHIS,wgtFile,opt)
  remap_z = ESMF_regrid_with_weights(z,wgtFile,opt)
   
  lat!0 = "lat"
  filevardef(fout, "lat", typeof(lat), getvardims(lat))
  fout->lat = (/lat/)

  lon!0 = "lon"
  filevardef(fout, "lon", typeof(lon), getvardims(lon))
  fout->lon = (/lon/)

  remap_ocnmask!0 = "lat"
  remap_ocnmask!1 = "lon"
  filevardef(fout, "oceanmask", typeof(remap_ocnmask), getvardims(remap_ocnmask))
  fout->oceanmask = (/remap_ocnmask/)

  remap_ocnfrac!0 = "lat"
  remap_ocnfrac!1 = "lon"
  filevardef(fout, "oceanfrac", typeof(remap_ocnfrac), getvardims(remap_ocnfrac))
  fout->oceanfrac = (/remap_ocnfrac/)
  
  remap_lndmask!0 = "lat"
  remap_lndmask!1 = "lon"
  filevardef(fout, "landmask", typeof(remap_lndmask), getvardims(remap_lndmask))
  fout->landmask = (/remap_lndmask/)

  remap_lndfrac!0 = "lat"
  remap_lndfrac!1 = "lon"
  filevardef(fout, "landfrac", typeof(remap_lndfrac), getvardims(remap_lndfrac))
  fout->landfrac = (/remap_lndfrac/)

  remap_depth!0 = "lat"
  remap_depth!1 = "lon"
  filevardef(fout, "depth", typeof(remap_depth), getvardims(remap_depth))
  fout->depth = (/remap_depth/)
  
  remap_height!0 = "lat"
  remap_height!1 = "lon"
  filevardef(fout, "height", typeof(remap_height), getvardims(remap_height))
  fout->height = (/remap_height/)
  
  remap_PHIS!0 = "lat"
  remap_PHIS!1 = "lon"
  filevardef(fout, "PHIS", typeof(remap_PHIS), getvardims(remap_PHIS))
  fout->PHIS = (/remap_PHIS/)

  remap_z!0 = "lat"
  remap_z!1 = "lon"
  filevardef(fout, "z", typeof(remap_z), getvardims(remap_z))
  fout->z = (/remap_z/)

end
