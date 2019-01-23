# -*- coding: utf-8 -*-

import os
import pdb
import argparse
import collections
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--mode',            type=str, required=True, help='train | test')
parser.add_argument('--train_dir',       type=str, help='directory to store training data', default='E:/face_data')
parser.add_argument('--test_dir',        type=str, help='directory to store testing data',  default='E:/NewBegining/Face0127/test')
parser.add_argument('--op', metavar='N', type=str, nargs='+', choices=['hanzitoidx', 'maxmin', 'extractlabel'], help='[hanzitoidx | maxmin | extractlabel]')
args = parser.parse_args()

class FaciexprDataloader():

    def __init__(self):
        self.mode      = args.mode
        self.test_dir  = args.test_dir
        self.train_dir = args.train_dir

    def dictionary(self):
        """生成key是汉字，value是数字的字典。
        """
        vocabulary = []
        folders = os.listdir(self.train_dir)
        for folder in folders:
            if os.path.isdir(os.path.join(self.train_dir, folder)):
                files = os.listdir(os.path.join(self.train_dir, folder))
                phoneme_files  = [file for file in files if file.find('_phoneme.txt') > 0]
                for phoneme_file in phoneme_files:
                    with open(os.path.join(self.train_dir, folder, phoneme_file), 'r', encoding='utf-8-sig') as f:
                        phoneme = f.readlines()
                    with open(os.path.join(self.train_dir, folder, phoneme_file).replace('phoneme', 'sentence'), 'r') as f:
                        sentence = f.readlines()
                    phoneme = [line.split() for line in phoneme]
                    from_phoneme = []
                    for line in phoneme[1:]:
                        if (line[-1] >= u'\u4e00' and line[-1] <= u'\u9fa5') or (line[-1] >= '0' and line[-1] <= '9'):
                            from_phoneme.append(line[-1])
                    from_sentence = [s for s in sentence[0] if s >= u'\u4e00' and s <= u'\u9fa5']
                    if from_phoneme == from_sentence:
                        for v in from_phoneme:
                            if v not in vocabulary:
                                vocabulary.extend(v)
        vocabulary.append('S') # S是silence
        vocabulary.append('UNK')
        vocab_to_idx = {vocab: idx for idx, vocab in enumerate(vocabulary, 1)}  # 0是padding
        print('vocab_to_idx: ', vocab_to_idx)
        print('length of vocab_to_idx: ', len(vocabulary))
        np.save('./data/vocab_to_idx.npy', vocab_to_idx)
        return vocab_to_idx

    @classmethod
    def hanzirepeat(cls, phoneme_file):
        """phoneme_file: phoneme的文件路径
        """
        new_sentence = []
        with open(phoneme_file, 'r', encoding='utf-8-sig') as f:
            phoneme = f.readlines()
        phoneme = [line.split() for line in phoneme]
        with open(phoneme_file.replace('phoneme', 'sentence'), 'r') as f:
            sentence = f.readlines()
        
        from_phoneme = []
        for line in phoneme[1:]:
            if (line[-1] >= u'\u4e00' and line[-1] <= u'\u9fa5') or (line[-1] >= '0' and line[-1] <= '9'):
                from_phoneme.append(line[-1])
        from_sentence = [s for s in sentence[0] if (s >= u'\u4e00' and s <= u'\u9fa5') or (s >= '0' and s <= '9')]
        
        if from_phoneme == from_sentence and from_sentence != []:
            for line in phoneme[1:]:
                if line[-1] == '<s>':
                    time_begin = float(line[0])
                    curr_hanzi = 'S'
                if line[0] == '0' and (line[-1] >= u'\u4e00' and line[-1] <= u'\u9fa5') or (line[-1] >= '0' and line[-1] <= '9'):
                    time_begin = float(line[0])
                    curr_hanzi = line[-1]
                if (line[-1] >= u'\u4e00' and line[-1] <= u'\u9fa5') or (line[-1] >= '0' and line[-1] <= '9'):
                    time_end   = float(line[0])
                    new_sentence.extend(curr_hanzi * round(30 * (time_end - time_begin + 1e-7)))
                    time_begin = float(line[0])
                    curr_hanzi = line[-1]
                if line[-1] == 'sil':
                    time_end   = float(line[0])
                    new_sentence.extend(curr_hanzi * round(30 * (time_end - time_begin + 1e-7)))
                    time_begin = float(line[0])
                    curr_hanzi = 'S'
                if line[-1] == '</s>':
                    time_end   = float(line[0])
                    new_sentence.extend(curr_hanzi * round(30 * (time_end - time_begin + 1e-7)))
                    time_begin = float(line[0])
                    time_end   = float(line[1])
                    new_sentence.extend('S' * round(30 * (time_end - time_begin + 1e-7)))
            modify_length = np.floor(30 * float(phoneme[-1][1]))
            if len(new_sentence) > modify_length:
                new_sentence = new_sentence[:int(modify_length)]
            elif len(new_sentence) < modify_length:
                add_length = modify_length - len(new_sentence)
                new_sentence.extend(['S'] * int(add_length))
        return new_sentence

    @classmethod
    def hanzitoidx(cls, phoneme_file, new_sentence, vocab_to_idx):
        """输入单个phoneme的文件路径和重复后的文本，将重复后的索引保存下来。
        phoneme_file: phoneme的文件路径
        new_sentence: hanzirepeat得到的结果
        """
        new_idx = [str(vocab_to_idx.get(hanzi, vocab_to_idx['UNK'])) for hanzi in new_sentence]
        with open(phoneme_file.replace('_phoneme', '_p_idx'), 'w', encoding='utf-8-sig') as f:
            f.write(' '.join(new_idx))

    def allhanzitoidx(self):
        
        # for training
        if self.mode == 'train':
            vocab_to_idx = self.dictionary()
            folders = os.listdir(self.train_dir)
            for folder in folders:
                if os.path.isdir(os.path.join(self.train_dir, folder)):
                    files = os.listdir(os.path.join(self.train_dir, folder))
                    phoneme_files = [file for file in files if file.find('_phoneme.txt') > 0]
                    for phoneme_file in phoneme_files:
                        if self.hanzirepeat(os.path.join(self.train_dir, folder, phoneme_file)) != []:
                            self.hanzitoidx(os.path.join(self.train_dir, folder, phoneme_file), self.hanzirepeat(os.path.join(self.train_dir, folder, phoneme_file)), vocab_to_idx)

        # for testing
        if self.mode == 'test':
            vocab_to_idx = np.load('./data/vocab_to_idx.npy').tolist()
            files = os.listdir(self.test_dir)
            phoneme_files = [file for file in files if file.find('_phoneme.txt') > 0]
            for phoneme_file in phoneme_files:
                if self.hanzirepeat(os.path.join(self.test_dir, phoneme_file)) != []:
                    self.hanzitoidx(os.path.join(self.test_dir, phoneme_file), self.hanzirepeat(os.path.join(self.test_dir, phoneme_file)), vocab_to_idx)

    @classmethod
    def extractlabel(cls, ble_or_ske):
        """提取单个blendshape中的6个元素到label（用于分类）、label_r（用于回归）中。
        ble_or_ske: open(skeleton.txt).readlines()
        """
        region = np.array([-0.01, 5., 20., 40., 60., 100.])
        label = []
        label_r = []
        for line in ble_or_ske:
            if line != '\n':
                line = line.split(' ')
                b = float(line[0]), float(line[2]), float(line[8]), float(line[14]), float(line[16]), float(line[17])
                b = list(b)
                label_r.append(str(b[0] / 100) + '_' + str(b[1] / 100) + '_' + str(b[2] / 100) + '_' + str(b[3] / 100) + '_' + str(b[4] / 100) + '_' + str(b[5] / 100))
                b[0], b[1], b[2], b[3], b[4], b[5] = \
                    pd.cut([b[0]], region, labels=False)[0], pd.cut([b[1]], region, labels=False)[0], pd.cut([b[2]], region, labels=False)[0], \
                    pd.cut([b[3]], region, labels=False)[0], pd.cut([b[4]], region, labels=False)[0], pd.cut([b[5]], region, labels=False)[0]
                label.append(str(b[0]) + '_' + str(b[1]) + '_' + str(b[2]) + '_' + str(b[3]) + '_' + str(b[4]) + '_' + str(b[5]))
        return label, label_r

    def allextractlabel(self):
        """directory: directory里放了很多个文件夹，文件夹里存放着p_idx.txt文件
        把所有重复后的索引保存到p_idx_f中；
        提取所有blendshape中的6个元素，保存到blendshape_f、r_blendshape_f中。
        """
        folders = os.listdir(self.train_dir)
        p_idx_f        = open('./data/p_idx.txt', 'w')
        blendshape_f   = open('./data/blendshape.txt', 'w')
        r_blendshape_f = open('./data/r_blendshape.txt', 'w')
        for folder in folders:
            if os.path.isdir(os.path.join(self.train_dir, folder)):
                files = os.listdir(os.path.join(self.train_dir, folder))
                p_idx_files = [file for file in files if file.find('_p_idx.txt') > 0]
                for p_idx_file in p_idx_files:
                    with open(os.path.join(self.train_dir, folder, p_idx_file), 'r', encoding='utf-8-sig') as f:
                        p_idx = f.read()
                        p_idx = p_idx.split()
                    if p_idx == []:
                        continue
                    blendshape_file = p_idx_file.replace('p_idx', 'blendshape')
                    with open(os.path.join(self.train_dir, folder, blendshape_file), 'r', encoding='utf-8-sig') as f:
                        blendshape = f.readlines()
                    length = len(blendshape) if len(blendshape) < len(p_idx) else len(p_idx)
                    if length == 0:
                        continue
                    p_idx, blendshape = p_idx[:length], blendshape[:length]
                    p_idx_f.write(' '.join(p_idx) + '\n')
                    label, label_r = self.extractlabel(blendshape)
                    blendshape_f.write(' '.join(label) + '\n')
                    r_blendshape_f.write(' '.join(label_r) + '\n')
        p_idx_f.close()
        blendshape_f.close()
        r_blendshape_f.close()

if __name__ == '__main__':
    dataloader = FaciexprDataloader()

    # C:/Users/lijianan/AppData/Local/Continuum/anaconda3/envs/pytorch0.4.1/python.exe e:/NewBegining/Face0127/FaciexprDataloader.py --op
    if 'hanzitoidx' in args.op:
        dataloader.allhanzitoidx()
    elif 'extractlabel' in args.op:
        dataloader.allextractlabel()
    else:
        raise Exception('There is no op matching with your choice.')