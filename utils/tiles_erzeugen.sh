gdal_translate -of COG \
  -co COMPRESS=LZW \
  Pilz_suitability.tif Pilz_suitability_cog.tif

gdal2tiles.py -z 8-15 Pilz_suitability_byte.tif tiles_suit/