import torch
import pdb

class Decoder:
    def __init__(self, lmap, ilmap):
        self.lmap = lmap
        self.ilmap = ilmap

    def to_string(self, indices):
        chars =[]
        indices = list(map(str, indices))
        # if '9'in indices:
        #     indices = ['10' if x == '9' else x for x in indices]
        for index in indices:
            # if index == '180':
            #     chars.append(' ')
            chars.append(self.ilmap[index])
        string = ''.join(chars)
        return string


    def decode(self, probs):
        """ Convert a probability matrix to sequences """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        max_probs = max_probs.squeeze()
        ls = max_probs.data.tolist()
        compressed = self.compress(ls)
        return self.to_string(compressed)

    def compress(self, values):
        result = []
        for i in range(1, len(values)):
            if values[i] != values[i-1]:
                result.append(values[i-1])
        result.append(values[len(values)-1])
        return result
