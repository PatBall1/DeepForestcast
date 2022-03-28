import os
import glob
import re
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show
from Download_functions import create_filenames, get_tiles

# Use same region name as download file
region = "Junin"

gridPath = "./inputs/gfc_tiles.shp"
boundaryPath = "./inputs/departments_amazon1_buffer0.09.shp"
boundPathTight = "./inputs/departments_amazon1.shp"
dwnldPath = "./downloads/" + region + "/"
outPath = "./outputs/" + region + "/"
if not os.path.exists(outPath):
    os.makedirs(outPath)

boundary = gpd.read_file(boundaryPath)
boundaryTight = gpd.read_file(boundPathTight)
tiles = get_tiles(boundary, gridPath)

types_static = ["treecover2000", "datamask", "gain"]
types_dynamic = ["lossyear", "last"]
types = types_static + types_dynamic
# Find years and file types available
files = glob.glob(dwnldPath + "/*.tif")
file_types = []
for file in files:
    year = [re.findall("[0-9]{4}", file)[0]]
    type = re.findall(r"(?=(" + "|".join(types) + r"))", file)
    #    print(year + type)
    file_types.append(year + type)

file_types = pd.DataFrame(file_types, columns=["year", "type"]).drop_duplicates()
print(file_types)

if os.path.exists("tmp.tif"):
    os.remove("tmp.tif")

# loop over file types and years
for index, row in file_types.iterrows():
    year = int(row["year"])
    type = row["type"]

    if len(glob.glob(outPath + "/" + type + "_" + str(year) + "*.tif")) > 0:
        print(type + "_" + str(year) + " already exists. Skipping...")
        continue
    else:
        #    year = 2019
        #    type = "last"
        names = create_filenames(tiles, year, type)
        print(names)

        files = []
        for name in names:
            path = dwnldPath + name
            src = rasterio.open(path)
            files.append(src)

        mosaic, out_trans = merge(files, bounds=boundary.geometry[0].bounds)

        mosaic = mosaic.astype("int16")

        # show(mosaic[0:3], transform=out_trans)
        out_profile = src.profile.copy()

        out_profile.update(
            {
                "dtype": "int16",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "nodata": -1,
            }
        )

        with rasterio.open("tmp.tif", "w", **out_profile) as tmp_out:
            tmp_out.write(mosaic)

        tmp = rasterio.open("tmp.tif")

        clip, out_trans = mask(tmp, boundary.geometry, nodata=-1, crop=True)
        #    show(clip[0:3], transform=out_trans)

        out_profile.update(
            {
                "count": 1,
                "height": clip.shape[1],
                "width": clip.shape[2],
                "transform": out_trans,
            }
        )

        # save image bands seperately
        if clip.shape[0] > 1:
            for i in range(clip.shape[0]):
                file_path_out = (
                    outPath + "/" + type + "_" + str(year) + "_" + str(i + 1) + ".tif"
                )
                rasterio.open(file_path_out, "w", **out_profile).write(
                    np.expand_dims(clip[i], 0)
                )
                print("Image band ", (i + 1), " saved")
        else:
            file_path_out = outPath + "/" + type + "_" + str(year) + ".tif"
            rasterio.open(file_path_out, "w", **out_profile).write(clip)

        [file.close() for file in files]
        src.close()
        tmp.close()
        os.remove("tmp.tif")

# For buffer tif
out_path = outPath + "buffer.tif"
if os.path.exists(out_path):
    print("Buffer already created. All done")
else:
    mask_path = glob.glob(outPath + "datamask*.tif")

    mask_layer = rasterio.open(mask_path[0])
    show(mask_layer.read(), transform=mask_layer.transform)

    mask_layer_dt = mask_layer.read()

    mask_layer_dt[mask_layer_dt == 0] = 1
    mask_layer_dt[mask_layer_dt == 2] = 1
    # mask_layer_dt[mask_layer_dt == -1] = 0

    out_profile = mask_layer.profile
    # out_profile.update({"nodata": 0})

    if os.path.exists("tmp.tif"):
        os.remove("tmp.tif")

    with rasterio.open("tmp.tif", "w", **out_profile) as bff_out:
        bff_out.write(mask_layer_dt)

    buffer = rasterio.open("tmp.tif")

    # Clip inside boundary
    inner, out_trans = mask(buffer, boundaryTight.geometry, nodata=-1)
    # inner, out_transI =  mask(buffer, boundaryTight.geometry, nodata=-999)

    buffer2 = buffer.read() + inner

    buffer2[buffer2 == -2] = -1
    buffer2[buffer2 == 0] = 1
    buffer2[buffer2 == 2] = 0

    show(buffer2, transform=buffer.transform)

    out_profile = buffer.profile
    out_profile.update(
        {
            "dtype": "int16",
            "height": buffer2.shape[1],
            "width": buffer2.shape[2],
            "transform": out_trans,
            "nodata": -1,
        }
    )

    rasterio.open(out_path, "w", **out_profile).write(buffer2)
    print("buffer file created:", os.path.exists(out_path))
    os.remove("tmp.tif")
"""
# %%
#  Check rasters came out right
import matplotlib
# %%
buffer = rasterio.open('../outputs/Madre_buffer.tif')
buffer.read()

show(buffer)
show(buffer.read()==1, transform=buffer.transform)
show(buffer.read()==0, transform=buffer.transform)
show(buffer.read()==-1, transform=buffer.transform)
# %%
tree2000 = rasterio.open('../outputs/Madre_treecover2000_2019.tif')
print(tree2000.read())
show(tree2000)
# %%
last = rasterio.open('../outputs/Madre_last_2019_1.tif')
print(last.read())
show(last)
"""
