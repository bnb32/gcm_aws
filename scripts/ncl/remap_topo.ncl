load "/usr/share/ncarg/nclscripts/csm/contributed.ncl"
load "/usr/share/ncarg/nclscripts/csm/popRemap.ncl"

begin

  fin = addfile(infile, "r")
  fout = addfile(outfile, "w")

  oceanmask = fin->mask
  oceanfrac = fin->frac

  printVarSummary(oceanmask)
  printVarSummary(oceanfrac)

  fout->mask = (/tointeger(PopLatLon(int2flt(1-oceanmask),"fv1.9x2.5","gx1v6","bilin","da","090206"))/)
  fout->frac = (/PopLatLon(1-oceanfrac,"fv1.9x2.5","gx1v6","bilin","da","090206")/)

end
