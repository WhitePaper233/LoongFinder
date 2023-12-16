# -*- coding: utf-8 -*-
import os
from PIL import Image
from multiprocessing import Pool

SRC_PATH = './datasets'
TARGET_PATH = './datasets'


def process_file(args):
    dataset_type, category, file = args
    file_path = os.path.join(SRC_PATH, dataset_type, category, file)
    save_path = os.path.join(TARGET_PATH, f'{dataset_type}_processed', category, file)

    image = Image.open(file_path)
    new_file = os.path.splitext(save_path)[0] + '.png'
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image.save(new_file)

    print(f"Processed file: {file}")


def process_dataset(dataset_type):
    print(f'----- Start to process {dataset_type} dataset -----')
    pool = Pool()
    for category in ['loong', 'notloong']:
        file_list = os.listdir(os.path.join(SRC_PATH, dataset_type, category))
        pool.map(process_file, [(dataset_type, category, file) for file in file_list])
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_dataset('train')
    process_dataset('val')
