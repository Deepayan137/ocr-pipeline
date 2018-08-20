import os
import re
import pdb
from tqdm import *
from utils.utils import *
from argparse import ArgumentParser
from parser.opts import base_opts

def parse_data(path):
	file_paths = list(map(lambda f: path + f, os.listdir(path)))
	def clean(base_name):
		return re.findall(r'\d+', base_name)[-1] 
	def read(text_file):
		with open(text_file, 'r') as f:
			text = f.read()
		return text
	content = {clean(os.path.basename(x)):read(x) for x in file_paths}
	return content

if __name__ == "__main__":
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	dir_ = args.path
	dirs = lambda f: os.path.join(dir_, f)
	folder_paths = map(dirs, ['Annotations/', 'Predictions/'])
	gt, pr = list(map(parse_data, folder_paths))
	keys = [key for key in gt.keys()]
	pbar = tqdm(keys)
	avgChar = AverageMeter("Character Accuracy")
	for key in pbar:
		try:
			pbar.set_description("Processing %s" % key)
			char_acc  = cer(pr[key], gt[key])
			avgChar.add(char_acc)
		except Exception as e:
			print('\n Key does not exist')
			print(key)
			print(e)
	print(avgChar.compute())