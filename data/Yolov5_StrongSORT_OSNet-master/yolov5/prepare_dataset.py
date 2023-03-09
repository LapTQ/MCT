import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

def extract_person_from_xml(xml_path):
    """
    :param xml_path: path to xml file
    :return:
        {
            'boxes': [
                {
                    'xmin': 12.0
                    'ymin':
                    'xmax':
                    'ymax':
                },
                {
                    ...
                }
            ],
            'filename': 'img_name.jpg',
            'size': {
                'width': 640,
                'height': 438,
                'depth': 3
            }
        }
    """
    root = ET.parse(xml_path).getroot()

    info = {}
    info['boxes'] = []

    for e in root:

        if e.tag == 'source':
            for e_sub in e:
                if e_sub.tag == 'url':
                    info['source'] = e_sub.text.split('/')[-2]

        if e.tag == 'filename':
            info['filename'] = e.text

        if e.tag == 'size':
            info['size'] = {}
            for e_sub in e:
                info['size'][e_sub.tag] = int(e_sub.text)

        if e.tag == 'object':
            box = {}
            is_person = False
            ########### TODO this code block is not smart
            for e_sub in e:
                if e_sub.tag == 'name' and e_sub.text == 'person':
                    is_person = True

            if not is_person: continue

            for e_sub in e:
                if e_sub.tag == 'bndbox':
                    for e_sub_sub in e_sub:
                        box[e_sub_sub.tag] = float(e_sub_sub.text)
                    info['boxes'].append(box)
            #################################################

    return info


def to_yolo_format(info):
    """
    Convert to yolo format
    :param info:
    :return: yolo format class_id, cx, cy, w, h normalized
    """
    buf = []
    for box in info['boxes']:
        cx = (box['xmin'] + box['xmax']) / (2 * info['size']['width'])
        cy = (box['ymin'] + box['ymax']) / (2 * info['size']['height'])
        w = (box['xmax'] - box['xmin']) / info['size']['width']
        h = (box['ymax'] - box['ymin']) / info['size']['height']

        buf.append('0 %.6f %.6f %.6f %.6f' % (cx, cy, w, h))

    return '\n'.join(buf)


def make_dataset(src_img_dir, src_lbl_dir, des_img_dir, des_lbl_dir):
    for file in tqdm(os.listdir(src_lbl_dir)):
        if not file.endswith('.xml'): continue

        info = extract_person_from_xml(os.path.join(src_lbl_dir, file))
        yolo_format = to_yolo_format(info)

        name, _ = os.path.splitext(file)
        print(yolo_format, file=open(os.path.join(des_lbl_dir, info['source'][:-4], name + '.txt'), 'w'))
        os.system(f"ln -s {os.path.join(src_img_dir, info['source'], name + '.jpg')} {os.path.join(des_img_dir, info['source'][:-4], name + '.jpg')}")



if __name__ == '__main__':

    shutil.rmtree('person_datasets', ignore_errors=True)

    src_img_dir = '/mnt/hdd3tb/Datasets/COCO2017'
    src_train_lbl_dir = '/mnt/hdd3tb/Datasets/COCO2017/train_coco/person'
    src_val_lbl_dir = '/mnt/hdd3tb/Datasets/COCO2017/val_coco/person'

    des_img_dir = 'person_datasets/images'
    des_lbl_dir = 'person_datasets/labels'

    os.makedirs(os.path.join(des_img_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(des_lbl_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(des_img_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(des_lbl_dir, 'val'), exist_ok=True)

    print('[INFO] Making training dataset')
    make_dataset(src_img_dir, src_train_lbl_dir, des_img_dir, des_lbl_dir)
    print('[INFO] Making validation dataset')
    make_dataset(src_img_dir, src_val_lbl_dir, des_img_dir, des_lbl_dir)

    with open('data/person_data.yaml', 'w') as f:
        f.write("""
train: person_datasets/images/train
val: person_datasets/images/val

# number of classes
nc: 1

# class name
names: ['person']
""")

# Laptq@123
