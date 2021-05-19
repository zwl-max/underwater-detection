import glob, tqdm, shutil, os
import random

random.seed(11)

if __name__ == '__main__':
    data_root = '../data/train/image'
    image_paths = glob.glob(data_root + '/*.jpg')
    
    # 划分数据集， 10%作为验证集
    train_size = int(len(image_paths) * 0.9)
    train_paths = image_paths[:train_size]
    val_paths   = image_paths[train_size:]
    
    os.makedirs('../data/train/train-image/', exist_ok=True)
    os.makedirs('../data/train/train-box/', exist_ok=True)
    os.makedirs('../data/train/val-image/', exist_ok=True)
    os.makedirs('../data/train/val-box/', exist_ok=True)
    
    for path in tqdm.tqdm(train_paths):
        base_name = os.path.basename(path)
        dst_name = os.path.join('../data/train/train-image', base_name)
        xml_name = base_name.split('.')[0] + '.xml'
        xml_src_path = os.path.join('../data/train/box', xml_name)
        xml_dst_path = os.path.join('../data/train/train-box', xml_name)
        shutil.copy(path, dst_name)
        shutil.copy(xml_src_path, xml_dst_path)
        
    for path in tqdm.tqdm(val_paths):
        base_name = os.path.basename(path)
        dst_name = os.path.join('../data/train/val-image', base_name)
        xml_name = base_name.split('.')[0] + '.xml'
        xml_src_path = os.path.join('../data/train/box', xml_name)
        xml_dst_path = os.path.join('../data/train/val-box', xml_name)
        shutil.copy(path, dst_name)
        shutil.copy(xml_src_path, xml_dst_path)