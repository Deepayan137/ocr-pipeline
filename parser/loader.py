import os
import cv2
import pdb
import numpy as np
from tqdm import *
import re
def image_prep(image_path):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)
	return thresh1

def parse_data(path):
	file_paths = list(map(lambda f: path + f, os.listdir(path)))
	file_paths = file_paths
	def clean(base_name):
		# return re.findall(r'\d+', base_name)[-1]
		return base_name.split('.')[0] 

	def read(text_file):
		with open(text_file, 'r') as f:
			text = f.read()
		return text
	if path.endswith('Images/'): 
		content = {clean(os.path.basename(x)):image_prep(x) for x in tqdm(file_paths)}
		return content
	content = {clean(os.path.basename(x)):read(x) for x in file_paths}
	return content

def images_and_truths(image, plot):
	def resizeImage(line_crop):
		try:
			height, width = np.shape(line_crop)
			if height is 0 or width is 0:
				line_crop = np.zeros((32, 32))
			height, width = np.shape(line_crop)
			ratio = 32/height
			resized_image = cv2.resize(line_crop, None, fx=ratio, fy=ratio, 
                interpolation = cv2.INTER_CUBIC)
			return np.array((resized_image), dtype='uint8')
		except Exception as e:
			print(e)
	def extract_units(unit):
		unit = list(map(lambda x:int(x), unit))
		x1, y1, w, h = unit
		x2 = x1+w; y2 = y1+h
		line_crop = image[int(y1):int(y2), int(x1):int(x2)]
		return line_crop

	li = plot.split()
	units = [li[i:i+4] for i in range(0, len(li), 4)]
	croppedImages = list(map(extract_units, units))
	unitImages = [resizeImage(croppedImages[i]) for i in range(len(croppedImages))]
	return unitImages

def read_book(**kwargs):
	Images = []
	book_path = kwargs["book_path"]
	dirs = lambda f: os.path.join(book_path, f)
	folder_paths = map(dirs, ['Images/', 'Segmentations/'])
	images, plots = list(map(parse_data, folder_paths))
	keys = [key for key in images.keys()]
	pdb.set_trace()
	pbar = tqdm(keys[:10])
	for key in pbar:	
		try:
			pbar.set_description("Processing %s" % key)
			unitImages = images_and_truths(images[key], plots[key])
			Images.append([key, unitImages])
		except Exception as e:
			print('\n Key does not exist')
			print(key)
			print(e)
	return Images