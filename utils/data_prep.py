import tqdm
import sys
import os
import glob
from subprocess import run
from shutil import copy, move
from tempfile import TemporaryDirectory
from multiprocessing import Pool
from datetime import datetime
from PIL import Image
import pdb
binary = '/home/deepayan/projects/batch_extract/j-layout'
source = '/ssd_scratch/cvit/deep/data/ndli/'
base_destination = '/ssd_scratch/cvit/deep/data/'
num_workers = 30
extensions = ['TIF', 'tif', 'tiff', 'TIFF', 'jpg', 'JPG', 'jpeg', 'JPEG']


def mute():
    sys.stdout = open(os.devnull, 'w')


def copy_annotations(image, destination):
    try:
        annotation = glob.glob(os.path.join(os.path.dirname(os.path.dirname(image)), 'Annotations', os.path.basename(image) + '.txt'))[0]
    except IndexError:
        with open('/home/deepayan/ocr-pipeline/error.log', 'a+') as error_log:
            error_log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -> ' + image + '\n')
        return False
    annotation_file_name = os.path.splitext(os.path.basename(image))[0] + '.txt'
    annotation_file_copy_path = os.path.dirname(annotation).replace(source, destination)
    if not os.path.isdir(annotation_file_copy_path):
            os.makedirs(annotation_file_copy_path, exist_ok=True)
    copy(annotation, os.path.join(annotation_file_copy_path, annotation_file_name))

    return True

def copy_image(image, destination):
    xml_path = os.path.dirname(os.path.dirname(image)) + '/' + 'META.XML'

    image_file_copy_path = os.path.join(os.path.dirname(os.path.dirname(image)), 'Images').replace(source, destination)
    if not os.path.isdir(image_file_copy_path):
            os.makedirs(image_file_copy_path, exist_ok=True)
    if os.path.exists(xml_path):
        copy(xml_path, os.path.dirname(os.path.dirname(image)).replace(source, destination))
    
    copy(image, image_file_copy_path)

    return True

def process_manual(image):
    destination = os.path.join(base_destination, 'Manual')

    if not copy_annotations(image, destination):
        return False
    copy_image(image, destination)
    image_name = os.path.basename(image)
    segmentation = glob.glob(os.path.join(os.path.dirname(os.path.dirname(image)), 'Segmentations', os.path.basename(image) + '.lines.txt'))[0]
    segmentation_file_name = os.path.basename(image) + '.lines.txt'
    segmentation_file_copy_path = os.path.join(os.path.dirname(os.path.dirname(image)), 'Segmentations').replace(source, destination)
    if not os.path.isdir(segmentation_file_copy_path):
            os.makedirs(segmentation_file_copy_path, exist_ok=True)
    copy(segmentation, os.path.join(segmentation_file_copy_path, segmentation_file_name))

    return True

def process_auto(image):
    destination = os.path.join(base_destination, 'ndli_Hindi/')

    # if not copy_annotations(image, destination):
    #     return False
    image_name = os.path.basename(image)
    book_name = image.split('/')[7]
    new_image_name = book_name+'_'+image_name
    new_image = os.path.join(os.path.dirname(image), new_image_name)
    copy(image, new_image)
    copy_image(new_image, destination)
    # print(image)
    # Run segmentation
    tmp_dir = TemporaryDirectory(prefix='dli_segmentation_')
    copy(binary, tmp_dir.name)
    copy(new_image, tmp_dir.name)
    os.chdir(tmp_dir.name)
    try:
        with open(os.devnull, 'w') as devnull:
            run(['./j-layout', os.path.basename(new_image)], stdout=devnull, stderr=devnull)

        # Copy Lines co-ordinates file from automatic segmentation
        lines_file_name = os.path.basename(new_image) + '.lines.txt'
        lines_file_copy_path = os.path.join(os.path.dirname(os.path.dirname(new_image)), 'Segmentations').replace(source, destination)
        if os.path.isfile(lines_file_name):
            if not os.path.isdir(lines_file_copy_path):
                os.makedirs(lines_file_copy_path, exist_ok=True)
            copy(os.path.join(tmp_dir.name, lines_file_name), lines_file_copy_path)
    except Exception as e:
        print(e)
        with open('/home/deepayan/ocr-pipeline/error.log', 'a+') as error_log:
            error_log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -> ' + image + '\n')
        return False
    return True

def data_prep():
    images = []
    for each in extensions:
        images += glob.glob(os.path.join(source, '**', '*.' + each), recursive=True)
    pool = Pool(num_workers)
    pdb.set_trace()
    # process_auto(images[0])
    # results = list(tqdm.tqdm(pool.imap(process_manual, images), total=len(images), ncols=120))
    results = list(tqdm.tqdm(pool.imap(process_auto, images), total=len(images), ncols=120))
    pool.close()
    pool.join()


if __name__ == '__main__':
    with open('error.log', 'w') as error_log:
        error_log.write('\n' * 2 + '*' * 80  + '\n' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -> ' + 'Starting Job\n' + '*' * 80 + '\n' * 3)
    data_prep()
