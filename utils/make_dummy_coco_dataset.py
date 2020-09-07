import os
import json



def create_ids_list(path):
    lista=[]
    for img_name in os.listdir(path):
        lista.append(int(img_name.replace('.jpg','')))
    return lista


def read_json(path):
    with open(path, 'r') as j:
        json_data = json.load(j)
    return json_data

def fit_json_to_dir(json_path,dir_path):
    l=create_ids_list(dir_path)
    json_data = read_json(json_path)
    json_data_new_images=[] ; json_data_new_annotations=[]
    for img in json_data['images']:
        if int(img['id']) in l:
            json_data_new_images.append(img)
    json_data['images'] = json_data_new_images
    for ann in json_data['annotations']:
        if int(ann['image_id']) in l:
            json_data_new_annotations.append(ann)
    json_data['annotations'] = json_data_new_annotations
    return json_data

def dump_json_to_file(json_df,path_to_dump):
    # to_json= json.dumps(json_df)
    with open(path_to_dump, 'w') as file:
        json.dump(json_df, file)

def main():
    json_data = fit_json_to_dir(json_path='/home/shalev/downloads/annotations/instances_val2017.json',
                            dir_path='/home/shalev/downloads/1pic_coco')
    dump_json_to_file(json_data,'/home/shalev/downloads/annotations/instances_val2017_1pic.json')

main()
