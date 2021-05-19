import glob, tqdm, shutil, os
import random

random.seed(11)

if __name__ == '__main__':
    data_root = '../data/train/image'
    image_paths = glob.glob(data_root + '/*.jpg')

    clean_paths = []
    u_paths =[]
    for path in image_paths:
        basename = os.path.basename(path)
        if basename[0] == 'c':
            clean_paths.append(path)
        else:
            u_paths.append(path)


    # 划分数据集， 从干净数据集中选10%作为验证集, 剩下的干净数据和噪声数据作为训练集
    val_size = int(len(image_paths) * 0.1)
    val_paths = clean_paths[:val_size]
    train_paths = clean_paths[val_size:] + u_paths

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