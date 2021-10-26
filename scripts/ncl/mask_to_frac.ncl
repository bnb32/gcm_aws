load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"
load "/usr/share/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
  
  fin = addfile("/data/cesm/inputdata/ocn/docn7/SOM/pop_frc.gx1v6.100513.nc", "rw")
 
  frac = int2dble(fin->mask)
  frac!0 = "nj"
  frac!1 = "ni"
  filevardef(fin, "frac", typeof(frac), getvardims(frac))
  fin->frac = (/frac/)

end
