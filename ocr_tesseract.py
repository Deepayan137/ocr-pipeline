from PIL import Image
import pytesseract
from argparse import ArgumentParser
import cv2
import os
import pdb
from parser.opts import base_opts
from utils.utils import *
from tqdm import *

def to_string(img, **kwargs):
	if kwargs['lang']:
		lang = kwargs['lang']
		return pytesseract.image_to_string(img, lang=lang)
	return pytesseract.image_to_string(img)



def read_image(imagename):
	try:
		image = cv2.imread(imagename)
	except:
		print('File Not Found')
	return image

def batch_ocr(dirname, savepath):
	 image_paths = list(map(lambda f: dirname +'/'+f , os.listdir(dirname)))
	 for path in tqdm(image_paths):
	 	image = read_image(path)
	 	imagename = os.path.basename(path)
	 	try:
		 	prediction = to_string(image, lang='san')
		 	filename = imagename + '.txt'
		 	save_text(prediction=prediction, filename=filename, 
		 		savedir=savepath)
	 	except Exception as e:
	 		print(imagename)
	 		print(e)
 		

if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	dir_ = args.path
	savepath = outdir('Predictions')
	gmkdir(savepath)


	basename = os.path.dirname(dir_)
	savepath = outdir(basename, 'Predictions')
	gmkdir(savepath)
	batch_ocr(dir_, savepath)
		