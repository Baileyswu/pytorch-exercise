import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from lang import Lang, prepareData
import random

SOS_token = 0
EOS_token = 1
 
class TorchDataset(Dataset):
    def __init__(self, lang_in :str, lang_out :str, reverse :bool, transform=None, repeat=1):
        '''
        :param 
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.lang_in, self.lang_out, self.pairs = prepareData(lang_in, lang_out, reverse)
        self.repeat = repeat
        self.len = len(self.pairs)
 
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        self.trans = transform
 
    def __getitem__(self, i):
        index = i % self.len
        unit = varFromPair(self.pairs[index], self.lang_in, self.lang_out)
        if self.trans != None:
            unit = self.data_preproccess(unit)
        return unit, self.pairs[index]
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.pairs) * self.repeat
        return data_len
 
    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.trans(data)
        return data

def indFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

def varFromSentence(lang, sentence):
    ind = indFromSentence(lang, sentence)
    ind.append(EOS_token)
    return torch.LongTensor(ind).view(-1, 1)

def varFromPair(pair, lang_in, lang_out):
    input_variable = varFromSentence(lang_in, pair[0])
    target_variable = varFromSentence(lang_out, pair[1])
    return (input_variable, target_variable)