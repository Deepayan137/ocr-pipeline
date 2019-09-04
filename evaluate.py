import os
import re
import pdb
from tqdm import *
from utils.utils import *
from argparse import ArgumentParser
from parser.opts import base_opts
from subprocess import run, PIPE
from tempfile import TemporaryDirectory

from jiwer import wer
import re
import string

def wer_evaluate(gt, prediction):
	tmp_dir = TemporaryDirectory(prefix='wer_')
	gt_file = open(os.path.join(tmp_dir.name, 'gt.txt'), 'w', encoding='utf-8')
	gt_file.write(gt)
	gt_file.close()
	prediction_file = open(os.path.join(tmp_dir.name, 'prediction.txt'), 'w', encoding='utf-8')
	prediction_file.write(prediction)
	prediction_file.close()
	WER_CLI = 'ocr-evaluation-tools/bin/ocrevalutf8'
	WER_CLI_PATH = 'ocr-evaluation-tools/bin'
	os.environ["PATH"] += ':' + WER_CLI_PATH
	output = run([WER_CLI, 'wordacc', gt_file.name, prediction_file.name], stdout=PIPE)
	msg = output.stdout.decode()
	string = msg.splitlines()[4].strip().split(' ')[0]
	if string == '------':
		accuracy = float(100)
	else:
		accuracy = float(string.rstrip('%'))
	return accuracy


def ndli_criteria_filter(text):
	text = re.sub('[' + string.punctuation + string.ascii_letters + ']', '', text)
	return text

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


def evaluate_book(path):
	dirs = lambda f: os.path.join(path, f)
	folder_paths = map(dirs, ['Annotations/', 'Predictions_CRNN/'])
	gt, pr = list(map(parse_data, folder_paths))
	keys = [key for key in gt.keys()]
	pbar = tqdm(keys)
	avgChar = AverageMeter("Book Character Accuracy")
	avg_wer = AverageMeter("Book Word Accuracy")
	ndli_avg_cer = AverageMeter("Book Character Accuracy - NDLI criteria")
	ndli_avg_wer = AverageMeter("Book Word Accuracy - NDLI criteria")
	for key in pbar:
		try:
			pbar.set_description("Processing %s" % key)
			char_acc  = cer(pr[key], gt[key])
			avgChar.add(char_acc)
			ndli_avg_cer.add(cer(ndli_criteria_filter(pr[key]), ndli_criteria_filter(gt[key])))
			# avg_wer.add(wer(gt[key].replace ( '\n', ' '), pr[key].replace ( '\n', ' ')))
			# ndli_avg_wer.add(wer(ndli_criteria_filter(gt[key].replace ( '\n', ' ')), ndli_criteria_filter(pr[key].replace ( '\n', ' '))))
			avg_wer.add(wer_evaluate(gt[key].replace ( '\n', ' '), pr[key].replace ( '\n', ' ')))
			ndli_avg_wer.add(wer_evaluate(ndli_criteria_filter(gt[key].replace ( '\n', ' ')), ndli_criteria_filter(pr[key].replace ( '\n', ' '))))
		except Exception as e:
			print('\n Key does not exist')
			print(key)
			print(e)

	return avgChar.compute(), avg_wer.compute(), ndli_avg_cer.compute(), ndli_avg_wer.compute()


if __name__ == "__main__":
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	dir_ = args.path
	accuracies = dict()
	CER = AverageMeter("Language Character Accuracy")
	WER = AverageMeter("Language Character Accuracy")
	NDLI_CER = AverageMeter("Language Character Accuracy")
	NDLI_WER = AverageMeter("Language Character Accuracy")
	import glob
	for book_path in glob.glob(os.path.join(dir_, '*')):
		result = evaluate_book(book_path)
		avg_cer, avg_wer, ndli_avg_cer, ndli_avg_wer = result
		# print(result)
		CER.add(avg_cer)
		WER.add(avg_wer)
		NDLI_CER.add(ndli_avg_cer)
		NDLI_WER.add(ndli_avg_wer)
		# if result:
		# 	accuracies[os.path.basename(book_path)] = result

	print('Average CER = ', CER.compute())
	print('Average WER = ', WER.compute())
	print('Average CER (NDLI criteria) = ', NDLI_CER.compute())
	print('Average WER (NDLI criteria) = ', NDLI_WER.compute())
	# import operator
	# sorted_accuracies = sorted(accuracies.items(), key=operator.itemgetter(1))
	# print(sorted_accuracies)
