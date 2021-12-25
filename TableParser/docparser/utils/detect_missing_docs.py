import os
import json

dataset_root = os.path.join('./', 'datasets')
train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/train')
VG_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/VG/train')
VG_img_dict_path =  os.path.join(VG_output_train_dataset, 'arxivdocs_weak_layout_train_scene_graph_image_data.json')
VG_obj_dict_path =  os.path.join(VG_output_train_dataset, 'arxivdocs_weak_layout_train_scene_graph_objects.json')
src_filenames = os.listdir(train_dataset)


print(len(src_filenames))

with open(VG_img_dict_path, 'r') as in_file:
    img_dict = json.load(in_file)
print(len(img_dict))


with open(VG_obj_dict_path, 'r') as in_file:
    obj_dict = json.load(in_file)

print('analyze json contents..')

img_dict_ids = dict()
for img_item in img_dict:
    img_dict_ids[img_item['image_id']] = img_item

obj_dict_ids = set()
for obj in obj_dict:
    obj_dict_ids.add(obj['image_id'])

print(len(img_dict_ids), len(obj_dict_ids))
print('looking form missing imgs')
missing_ids = set(img_dict_ids.keys()) - obj_dict_ids
print('missing: {}'.format( missing_ids))
