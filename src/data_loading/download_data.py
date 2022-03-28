import os
from Download_functions import download_data, import_boundary

# os.chdir('..')
os.getcwd()

# Define a name for the region
region = "Junin"

# Import boundary data
boundaryPath = "./inputs/departments_amazon1.shp"
buffer = import_boundary(boundaryPath, buffer=0.09)

# Import global grid
gridPath = "./inputs/gfc_tiles.shp"
# grid = gpd.read_file(gridPath)
dwnldPath = "./downloads"
dwnldPath = dwnldPath + "/" + region

years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
# years = [2014, 2018, 2019]
# types can take values...
# define static and dynamic layers seperately to reduce amount of data to download

types_static = ["treecover2000", "datamask", "gain"]
types_dynamic = ["lossyear", "last"]

download_data(buffer, dwnldPath, years, types_static, types_dynamic, gridPath)
