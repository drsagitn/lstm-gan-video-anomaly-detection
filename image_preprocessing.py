from PIL import Image
import os

base_width = 115
target_folder = 'data/UCSDped1/Train'
out_folder = 'data/scaled_data/UCSDped1/Train'

for d in os.listdir(target_folder):
    image_folder_path = os.path.join(target_folder, d)
    os.mkdir(os.path.join(out_folder, d))
    for image_file_name in os.listdir(image_folder_path):
        image_file_path = os.path.join(image_folder_path, image_file_name)
        print("Processing ", image_file_path)
        img = Image.open(image_file_path)
        percent = base_width/float(img.size[0])
        hsize = int(float(img.size[1])*percent)
        img = img.resize((base_width, hsize), Image.ANTIALIAS)
        img.save(os.path.join(out_folder, d, image_file_name))
