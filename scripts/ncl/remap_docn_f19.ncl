load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"
load "/usr/share/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
  
  srcGridFile="/data/bnb32/inputdata/mapping_files/tmp_src.nc"
  dstGridFile="/data/bnb32/inputdata/mapping_files/tmp_dst.nc"
  wgtFile="/data/bnb32/inputdata/mapping_files/tmp_wgts.nc"

  ocn_dom = addfile("/data/cesm/inputdata/share/domains/domain.ocn.1.9x2.5_gx1v6_090403.nc","r")
  fin = addfile("/data/cesm/inputdata/ocn/docn7/SOM/pop_frc.1x1d.090130.nc", "rw")
  fout = addfile("/data/cesm/inputdata/ocn/docn7/SOM/pop_frc.f19.nc", "cw")
  
  dimNames = (/"time","lat", "lon"/)
  dimSizes = (/12, 96, 144/)
  dimUnlim = (/False, False, False/)
  filedimdef(fout, dimNames, dimSizes, dimUnlim)
  
  opt = True
  opt@ForceOverwrite = True
  opt@GridMask = new((/180,360/),integer,1)
  
  lat  = latGlobeF(180, "lat", "latitude", "degrees_north")
  lon  = lonGlobeF(360, "lon", "longitude", "degrees_east")
  
  rectilinear_to_SCRIP(srcGridFile,lat,lon,opt)
  
  delete(opt)
  delete(lat)
  delete(lon)
  opt = True
  opt@ForceOverwrite = True
  opt@Check = True
  opt@GridMask = new((/96,144/),integer,1)
  
  yc  = latGlobeF(96, "lat", "latitude", "degrees_north")
  xc  = lonGlobeF(144, "lon", "longitude", "degrees_east")
      
  rectilinear_to_SCRIP(dstGridFile,yc,xc,opt)
  
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

  ;remap_area = new((/96,144/),double,0)
  ;remap_mask = new((/96,144/),integer,0)
  remap_S = new((/12,96,144/),float,0)
  remap_T = new((/12,96,144/),float,0)
  remap_U = new((/12,96,144/),float,0)
  remap_V = new((/12,96,144/),float,0)
  remap_dhdx = new((/12,96,144/),float,0)
  remap_dhdy = new((/12,96,144/),float,0)
  remap_hblt = new((/12,96,144/),float,0)
  remap_qdp = new((/12,96,144/),float,0)
  
  remap_area = ocn_dom->area ;ESMF_regrid_with_weights(fin->area,wgtFile,opt) 
  remap_mask = ocn_dom->mask ;tointeger(ESMF_regrid_with_weights(int2flt(fin->mask),wgtFile,opt)) 
  
  do t=0,11
    remap_S(t,:,:) = ESMF_regrid_with_weights(fin->S(t,:,:),wgtFile,opt) 
    remap_T(t,:,:) = ESMF_regrid_with_weights(fin->T(t,:,:),wgtFile,opt) 
    remap_U(t,:,:) = ESMF_regrid_with_weights(fin->U(t,:,:),wgtFile,opt) 
    remap_V(t,:,:) = ESMF_regrid_with_weights(fin->V(t,:,:),wgtFile,opt) 
    remap_dhdx(t,:,:) = ESMF_regrid_with_weights(fin->dhdx(t,:,:),wgtFile,opt) 
    remap_dhdy(t,:,:) = ESMF_regrid_with_weights(fin->dhdy(t,:,:),wgtFile,opt) 
    remap_hblt(t,:,:) = ESMF_regrid_with_weights(fin->hblt(t,:,:),wgtFile,opt) 
    remap_qdp(t,:,:) = ESMF_regrid_with_weights(fin->qdp(t,:,:),wgtFile,opt) 
  end do

  replace_ieeenan(remap_S,0.0,0)
  replace_ieeenan(remap_U,0.0,0)
  replace_ieeenan(remap_V,0.0,0)
  replace_ieeenan(remap_T,0.0,0)
  replace_ieeenan(remap_dhdx,0.0,0)
  replace_ieeenan(remap_dhdy,0.0,0)
  replace_ieeenan(remap_hblt,0.0,0)
  replace_ieeenan(remap_qdp,0.0,0)

  yc!0 = "lat"
  filevardef(fout, "yc", typeof(yc), getvardims(yc))
  filevarattdef(fout, "yc", fin->yc)
  fout->yc = (/yc/)
  
  xc!0 = "lon"
  filevardef(fout, "xc", typeof(xc), getvardims(xc))
  filevarattdef(fout, "xc", fin->xc)
  fout->xc = (/xc/)

  time = fin->time
  time!0 = "time"
  filevardef(fout, "time", typeof(time), getvardims(time))
  filevarattdef(fout, "time", fin->time)
  fout->time = (/time/)
  
  remap_mask!0 = "lat"
  remap_mask!1 = "lon"
  filevardef(fout, "mask", typeof(fin->mask), getvardims(fin->mask))
  filevarattdef(fout, "mask", ocn_dom->mask)
  fout->mask = (/ocn_dom->mask/)

  remap_area!0 = "lat"
  remap_area!1 = "lon"
  filevardef(fout, "area", typeof(fin->area), getvardims(fin->area))
  filevarattdef(fout, "area", ocn_dom->area)
  fout->area = (/ocn_dom->area/)

  remap_S!0 = "time"
  remap_S!1 = "lat"
  remap_S!2 = "lon"
  filevardef(fout, "S", typeof(remap_S), getvardims(remap_S))
  filevarattdef(fout, "S", fin->S)
  fout->S = (/remap_S/)

  remap_T!0 = "time"
  remap_T!1 = "lat"
  remap_T!2 = "lon"
  filevardef(fout, "T", typeof(remap_T), getvardims(remap_T))
  filevarattdef(fout, "T", fin->T)
  fout->T = (/remap_T/)

  remap_U!0 = "time"
  remap_U!1 = "lat"
  remap_U!2 = "lon"
  filevardef(fout, "U", typeof(remap_U), getvardims(remap_U))
  filevarattdef(fout, "U", fin->U)
  fout->U = (/remap_U/)

  remap_V!0 = "time"
  remap_V!1 = "lat"
  remap_V!2 = "lon"
  filevardef(fout, "V", typeof(remap_V), getvardims(remap_V))
  filevarattdef(fout, "V", fin->V)
  fout->V = (/remap_V/)

  remap_dhdx!0 = "time"
  remap_dhdx!1 = "lat"
  remap_dhdx!2 = "lon"
  filevardef(fout, "dhdx", typeof(remap_dhdx), getvardims(remap_dhdx))
  filevarattdef(fout, "dhdx", fin->dhdx)
  fout->dhdx = (/remap_dhdx/)

  remap_dhdy!0 = "time"
  remap_dhdy!1 = "lat"
  remap_dhdy!2 = "lon"
  filevardef(fout, "dhdy", typeof(remap_dhdy), getvardims(remap_dhdy))
  filevarattdef(fout, "dhdy", fin->dhdy)
  fout->dhdy = (/remap_dhdy/)

  remap_hblt!0 = "time"
  remap_hblt!1 = "lat"
  remap_hblt!2 = "lon"
  filevardef(fout, "hblt", typeof(remap_hblt), getvardims(remap_hblt))
  filevarattdef(fout, "hblt", fin->hblt)
  fout->hblt = (/remap_hblt/)

  remap_qdp!0 = "time"
  remap_qdp!1 = "lat"
  remap_qdp!2 = "lon"
  filevardef(fout, "qdp", typeof(remap_qdp), getvardims(remap_qdp))
  filevarattdef(fout, "qdp", fin->qdp)
  fout->qdp = (/remap_qdp/)

end
