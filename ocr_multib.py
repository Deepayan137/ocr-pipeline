import os
import sys
import torch
from torch import nn, optim
from torch.autograd import Variable
from tqdm import *
from parser.opts import *
from argparse import ArgumentParser
from parser.loader import *
# from parser import read_book
from utils.utils import *
from model.model import *

def to_text(page, model, vocab):
	
	def unit_ocr(sequence):
		model.eval()
		sequence = sequence.view(1, *sequence.size())
		sequence = Variable(sequence, requires_grad=False)
		output = model(sequence)
		output = output.contiguous()
		prediction = get_prediction(output, vocab)

		return prediction
	
	lines = [unit_ocr(sequence) for sequence in page]
	# pdb.set_trace()
	return '\n'.join(lines)
	

def batch_ocr(**kwargs):
	opt._parse(kwargs)
	path = opt.path
	lang = opt.lang
	dest = opt.dest
	lookup_filename = os.path.join('lookups', '%s.vocab.json'%lang)
	vocab = load_vocab(lookup_filename)
	lmap, ilmap = vocab['v2i'], vocab['i2v']
	nclasses = len(lmap)
	# pdb.set_trace()
	pretrained = 'models/%s_%s.tar'%(lang, opt.type_)
	print(pretrained)
	if opt.type_ == 'BLSTM':
		model = GravesNet(opt.imgH, opt.hidden_size, nclasses, opt.depth)
	elif opt.type_ == 'CRNN':
		model = CRNN(opt.imgH, opt.nchannels, nclasses,opt.hidden_size, stn_flag=False)
	# pdb.set_trace()
	cuda = torch.cuda.is_available()
	if cuda:
		model = model.cuda()
		# model = nn.DataParallel(model)
	if os.path.isfile(pretrained):
		with open(pretrained, "rb") as model_file:
			print('loaded file')
			checkpoint = torch.load(model_file)
			model.load_state_dict(checkpoint['state_dict'])
	f = lambda x: path + '/'+ x
	book_paths = list(map(f, os.listdir(path)))
	for i, book_path in enumerate(book_paths):
		# pdb.set_trace()
		book_name = os.path.basename(book_path)
		# savepath = os.path.join(dest, book_name, 'Predictions_CRNN')
		
		savepath = outdir(book_path,'Predictions_CRNN')
		with open('record.txt', 'a') as f:
			f.write('[%d]/[%d] %s\n'%(i, len(book_paths), book_path))
		print('{}, {}'.format(i, book_path))
		if not os.path.isdir(savepath):
			os.makedirs(savepath)
			pages = read_book(book_path=book_path)
			print('writing')
			for filename, images in tqdm(pages):
			# for key in tqdm(pages.keys()):
				# images = pad_seq(images)
				# images = pages[key]
				images = list(map(gpu_format, images))
				prediction = to_text(images, model, vocab)
				# pdb.set_trace()
				save_text(prediction=prediction, filename='%s.txt'%filename, 
			 			savedir=savepath)
		else:
			print('Already there')
			


if __name__ == '__main__':
	import fire
	fire.Fire()
	