###
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "./data/traing_image_500Image/train"

# set path for coco json to be saved
save_json_path = "./data/traing_image_500Image/train.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)