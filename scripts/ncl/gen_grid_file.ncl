load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"
load "/usr/share/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
  
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new((/96,144/),integer,1)
  gridFile = "/data/bnb32/inputdata/grid_files/grid_1.9x2.5.nc"
  
  lat  = latGlobeF(96, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(144, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(gridFile,lat,lon,opt)
  
  delete(opt)
  delete(gridFile)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new((/180,360/),integer,1)
  gridFile = "/data/bnb32/inputdata/grid_files/grid_1x1.nc"
  
  lat  = latGlobeF(180, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(360, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(gridFile,lat,lon,opt)

  
  delete(opt)
  delete(gridFile)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new((/384,320/),integer,1)
  gridFile = "/data/bnb32/inputdata/grid_files/grid_gx1v6.nc"
  
  lat  = latGlobeF(384, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(320, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(gridFile,lat,lon,opt)

end  
