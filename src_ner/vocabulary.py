import codecs
import random

import numpy as np
import collections

import sklearn.preprocessing

import utils_re
import utils
import brat2conll
import en_core_web_sm




class Vocabulary(object):
    expressions = [
        r"[A-Z]",
        r"[0-9]{10,12}",
        r"[0-9]+",
        r"(?:[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])+",
        r"([a-zA-Z0-9][a-zA-Z0-9_.+-]*@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)",
        r"(?:ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        r"[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+",
        r"(?:[0-9]+\.)+[0-9]+",
        r"(?:[0-9]+\.){3}[0-9]+",
        r"(?:[0-9]+\.){5}[0-9]+",
        r"(?:AS|as)[0-9]+",
        r"(?:CVE|exploit)",
        r"(?:HK|hk)[a-zA-Z0-9/]+"]
    def __init__(self,token_to_vector,embedding_dim,expressions = None):
        self.token2index ={}
        self.tokens = []
        self.characters = []
        self.character2index = {}
        self.UNK = 'UNK'
        self.UNK_INDEX = 0
        self.tokens.append(self.UNK)
        self.token2index[self.UNK] = 0
        self.PADDING_INDEX = 0
        self.characters.append('pad')
        # self.token2vector = token_to_vector
        self.dictionary = list(token_to_vector.keys())
        # self.token2vector[self.UNK] = np.zeros([embedding_dim])
        self.embedding_dim = embedding_dim
        self.label2index = {}
        self.labels = []
        self.nlp = en_core_web_sm.load()
        self.verbose = False
        self.number_of_classes = 1

        if expressions != None:
            self.expressions = expressions
        self.size_pattern_vector = len(self.expressions)


    def add_token(self,token):
        lower = token.lower()
        if lower in self.dictionary:
            if lower not in self.token2index.keys():
                self.token2index[lower] = len(self.tokens)
                self.tokens.append(lower)
                self.vocabulary_size = len(self.tokens)

        for char in list(token):
            self.add_character(char)


    def add_character(self,char):
        if char not in self.character2index.keys():
            self.character2index[char]=len(self.characters)
            self.characters.append(char)
            self.alphabet_size = len(self.characters)

    def add_label(self,label):
        if label not in self.label2index.keys():
            self.label2index[label]=len(self.label2index)
            self.labels.append(label)
            self.number_of_classes = len(self.labels)

    def get_embedding(self,token2vector,embedding_dim):
        unk_vector = np.zeros([embedding_dim])
        return [token2vector.get(token,unk_vector) for token in self.tokens]



    def transform(self,tokens,labels=None):
        for sequence in tokens:
            for token in sequence:
                self.add_token(token)
        pattern = [[utils_re.get_pattern(token,self.expressions) for token in sequence] for sequence in tokens]
        token_indices = []
        characters = []
        character_indices = []
        token_lengths = []
        character_indices_padded = []
        for token_sequence in tokens:
            token_indices.append(
                [self.token2index.get(token.lower(), self.UNK_INDEX) for token in token_sequence])
            characters.append([list(token) for token in token_sequence])
            character_indices.append([[self.character2index.get(character,0) for character in token] for token in token_sequence])
            token_lengths.append([len(token) for token in token_sequence])
            longest_token_length_in_sequence = max(token_lengths[-1])
            character_indices_padded.append(
                [utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_INDEX)
                 for temp_token_indices in character_indices[-1]])

        if labels == None:
            return token_indices, character_indices_padded, token_lengths, pattern
        label_indices = []
        for label_sequence in labels:
            for label in label_sequence:
                self.add_label(label)
        for label_sequence in labels:
            label_indices.append([self.label2index[label] for label in label_sequence])


        if self.verbose:
            print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths[0][0:10]))
        if self.verbose:
            print('characters[\'train\'][0][0:10]: {0}'.format(characters[0][0:10]))
        if self.verbose:
            print('token_indices[\'train\'][0:10]: {0}'.format(token_indices[0:10]))
        if self.verbose:
            print('label_indices[\'train\'][0:10]: {0}'.format(label_indices[0:10]))
        if self.verbose:
            print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices[0][0:10]))
        if self.verbose:
            print('character_indices_padded[\'train\'][0][0:10]: {0}'.format(
                character_indices_padded[0][0:10]))  # Vectorize the labels
        # [Numpy 1-hot array](http://stackoverflow.com/a/42263603/395857)
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(len(self.labels) + 1))


        label_vector_indices = []
        for label_indices_sequence in label_indices:
            label_vector_indices.append(label_binarizer.transform(label_indices_sequence))


        if self.verbose:
            print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))
        if self.verbose:
            print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))

        return token_indices, character_indices_padded, token_lengths, pattern , label_indices, label_vector_indices
    def transform_text(self,text):
        text = text + '.'
        sentences = brat2conll.get_sentences_and_tokens_from_spacy(text, self.nlp)
        sentences = [[token['text'] for token in sentence] for sentence in sentences]
        return self.transform(sentences)

    def parse_conll(self,pathfile):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        line_count = -1
        tokens = []
        labels = []
        # patterns = []
        new_token_sequence = []
        new_label_sequence = []
        # new_pattern_sequence = []
        if pathfile:
            f = codecs.open(pathfile, 'r', 'UTF-8')
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels.append(new_label_sequence)
                        tokens.append(new_token_sequence)
                        # patterns.append(new_pattern_sequence)
                        new_token_sequence = []
                        new_label_sequence = []
                        # new_pattern_sequence = []
                    continue
                token = str(line[0])
                label = str(line[-1])
                # pattern = utils_re.get_pattern(token)
                token_count[token] += 1
                label_count[label] += 1

                new_token_sequence.append(token)
                new_label_sequence.append(label)
                # new_pattern_sequence.append(pattern)

                for character in token:
                    character_count[character] += 1

                # if self.debug and line_count > 200: break  # for debugging purposes

            if len(new_token_sequence) > 0:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
                # patterns.append(new_pattern_sequence)
            f.close()
        return self.transform(tokens,labels)



