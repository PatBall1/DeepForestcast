import time
import os
from PIL import Image
from process_functions import create_tnsors_pixels

Image.MAX_IMAGE_PIXELS = None

region = "Junin"
last_yr = 20
sourcepath = "./outputs/" + region
wherepath = "../../data/" + region

if not os.path.exists(wherepath):
    os.makedirs(wherepath)

for year in range(14, last_yr + 1):
    start = time.time()
    print(year)
    static, image, pixels, labels = create_tnsors_pixels(
        year=year,
        latest_yr=last_yr,
        tree_p=30,
        cutoffs=[2, 5, 8],
        sourcepath=sourcepath,
        rescale=True,
        wherepath=wherepath,
    )
    print("Total time (in seconds) needed to create tensors: ", time.time() - start)
