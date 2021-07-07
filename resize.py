import os
import glob
from tqdm import tqdm #just for progress bar
from PIL import Image, ImageFile #python imaging library to manipulate images
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample = Image.BILINEAR
    )

    img.save(outpath)

#resize the train images

input_folder = "C:/Users/shravan/Pictures/Camera Roll"
output_folder = r"C:\Users\shravan\Documents\project\test"
images = glob.glob(os.path.join(input_folder, "*.jpg")) 
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512,512)
    ) for i in tqdm (images)
)