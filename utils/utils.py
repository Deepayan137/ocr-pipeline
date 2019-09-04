import os
import cv2
import torch
import numpy as np
import json 
import Levenshtein as lev
from utils.coding import *

def outdir(*names):
    # base_dir = '/home/deepayan/ocr-pipeline/'
    return os.path.join(*names)


def gmkdir(path):
    flag = False
    if not os.path.exists(path):
        os.makedirs(path)
        flag = True
    return flag
    

def cer(predictions, truths):
    sum_edit_dists = lev.distance(predictions, truths)
    sum_gt_lengths = sum(map(len, truths))
    fraction = sum_edit_dists/sum_gt_lengths
    percent = fraction*100
    return (100.0-percent)

def load_vocab(vocab_file):
    with open(os.path.join(vocab_file), 'r') as file:
        vocab = json.load(file)
        return vocab

def codebook(filename):
    keys = open(filename).read().splitlines()
    labels = list(map(fmt_to_unicode, keys))

    # Add the blank character.
    labels.insert(0, '')
    indices = list(range(len(labels)))

    lmap = dict(zip(labels, indices))
    invlmap = dict(zip(indices, labels))
    return (lmap, invlmap)


def fmt_to_unicode(fmt):
    codepoint_repr = fmt[1:]
    codepoint_value = int(codepoint_repr, 16)
    return chr(codepoint_value)

def gpu_format(sequence):
    sequence = torch.Tensor(np.array(sequence, dtype=np.float32))
    sequence = sequence.unsqueeze(0)
    if torch.cuda.is_available():
        sequence = sequence.cuda()
    return sequence

def get_prediction(prediction, vocab):
    lmap, ilmap = vocab['v2i'], vocab['i2v']
    decoder  = Decoder(lmap, ilmap)
    
    prediction =  decoder.decode(prediction)
    return prediction

def pad_seq(sequences):
    padded = []
    widths = [np.shape(seq)[1] for seq in sequences]
    max_width = int(np.mean(widths))
    print("Max width: %d"%max_width)
    def pad(seq):
        diff = abs(max_width - np.shape(seq)[1])
        padding = np.ones((32, diff), dtype='uint8')
        padded_image = np.concatenate((seq, padding), axis=1)
        return (padded_image)
    for i, seq in enumerate(sequences):
        if np.shape(seq)[1] < max_width:
            padded.append(pad(seq))
        else:
            resized_image = cv2.resize(seq, (max_width, 32),
                        interpolation=cv2.INTER_AREA)
            padded.append((resized_image))
    return padded

def save_text(**kwargs):
    text = kwargs['prediction']
    filename = kwargs['filename']
    if kwargs['savedir']:
        savedir = kwargs['savedir']
        filename = os.path.join(savedir, filename)
    with open(filename, 'w') as f:
        f.write(text)
class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1*float("inf")
        self.min = float("inf")

    def add(self, element):
        # pdb.set_trace()
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        # pdb.set_trace()
        if self.count == 0:
            return float("inf")
        return self.total/self.count
    
    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)"%(self.name, self.min, self.compute(), self.max)