import os
import sys
import torch
from torch import nn, optim
from torch.autograd import Variable
from tqdm import *
from parser.opts import *
from argparse import ArgumentParser
from parser.loader import *
from utils.utils import *
from model.model import *

def to_text(page, model, vocab):
	def unit_ocr(sequence):
		sequence = sequence.view(1, *sequence.size())
		sequence = Variable(sequence, requires_grad=False)
		output = model(sequence)
		output = output.contiguous()
		prediction = get_prediction(output, vocab)
		return prediction
	lines = [unit_ocr(sequence) for sequence in page]
	return '\n'.join(lines)

def batch_ocr(**kwargs):
	opt._parse(kwargs)
	path = opt.path
	lang = opt.lang
	
	lookup_filename = os.path.join('lookups', '%s.vocab.json'%lang)
	vocab = load_vocab(lookup_filename)
	lmap, ilmap = vocab['v2i'], vocab['i2v']
	nclasses = len(lmap)
	pretrained = 'models/%s_%s.tar'%(lang, opt.type_)

	if opt.type_ == 'BLSTM':
		model = GravesNet(opt.imgH, opt.hidden_size, nclasses, opt.depth)
	elif opt.type_ == 'CRNN':
		model = CRNN(opt.imgH, opt.imgW, opt.nchannels, nclasses,opt.hidden_size, stn_flag=False)

	cuda = torch.cuda.is_available()
	if cuda:
		model = model.cuda()

	if os.path.isfile(pretrained):
		with open(pretrained, "rb") as model_file:
			checkpoint = torch.load(model_file)
			model.load_state_dict(checkpoint['state_dict'])

	savepath = outdir(path,'Predictions_CRNN')
	gmkdir(savepath)
	pages = read_book(book_path=path)
	print('writing')
	for filename, images in tqdm(pages):
		# images = pad_seq(images)
		images = list(map(gpu_format, images))
		prediction = to_text(images, model, vocab)
		save_text(prediction=prediction, filename='%s.txt'%filename, 
		 		savedir=savepath)

if __name__ == '__main__':
	import fire
	fire.Fire()
	