# %%
#!/usr/bin/python
from tqdm import tqdm
import glob
import os
import shutil
depth_dirs = glob.glob("./dataset_prepare/kitti/val/*/proj_depth/groundtruth/image_02")
for dir in tqdm(depth_dirs):
    new_depth_dir = "./data/kitti/" + dir.split("/")[-4]+"/depth_gt"
    new_image_dir = "./data/kitti/" + dir.split("/")[-4]+"/rgb"
    os.makedirs(new_depth_dir, exist_ok=True)
    os.makedirs(new_image_dir, exist_ok=True)
    for depth_file in sorted(glob.glob(dir + "/*.png"))[:110]: #../data/kitti/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png
        new_path = new_depth_dir + "/" + depth_file.split("/")[-1]
        shutil.copy(depth_file, new_path)
        mid = "_".join(depth_file.split("/")[4].split("_")[:3])
        image_file = depth_file.replace('val', mid).replace('proj_depth/groundtruth/image_02', 'image_02/data')
        if os.path.exists(image_file):
            new_path = new_image_dir + "/" + image_file.split("/")[-1]
            shutil.copy(image_file, new_path)
        else:
            print("Image file does not exist: ", image_file)