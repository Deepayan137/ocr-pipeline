import os
import subprocess
import tempfile
from shutil import copy, rmtree
import sys
import pdb
def run_segmentation(image_path):
    """ Run segmentation and produce plot file and plot image """

    abs_path = '/home/deepayan/projects/batch_extract/'
    script_name = abs_path + 'j-layout'
    tmp_dir = abs_path + 'tmp/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    image_name = os.path.split(image_path)[-1]
    try:
        copy(image_path, tmp_dir)
        tmp_image = os.path.join(tmp_dir, image_name)
    except AttributeError:
        return 1

    plot_file = os.path.join(tmp_dir, image_name + '.lines.txt')
    cmd = [script_name, tmp_image, plot_file]
    subprocess.run(cmd)
    with open(plot_file, 'r') as f:
        plot = f.read()
    rmtree(tmp_dir)
    return plot
