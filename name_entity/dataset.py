import codecs
import glob

import os

import numpy as np
import collections

import sklearn.preprocessing

import utils_re
import utils
import en_core_web_sm

import utils_nlp



class Dataset(object):
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

    def __init__(self, expressions=None):
        self.token2index = {}
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
        # self.dictionary = (token_to_vector.keys())
        # self.token2vector[self.UNK] = np.zeros([embedding_dim])

        # self.embedding_dim = embedding_dim
        self.label2index = {}
        self.labels = []
        # self.nlp = en_core_web_sm.load()
        self.verbose = False
        self.number_of_classes = 1

        if expressions != None:
            self.expressions = expressions[:]
        self.size_pattern_vector = len(self.expressions)

    def add_token(self, token ,token_to_vector):
        lower = token.lower()
        if lower in token_to_vector.keys():
            if lower not in self.token2index.keys():
                self.token2index[lower] = len(self.tokens)
                self.tokens.append(lower)
                self.vocabulary_size = len(self.tokens)

        for char in list(token):
            self.add_character(char)

    def add_character(self, char):
        if char not in self.character2index.keys():
            self.character2index[char] = len(self.characters)
            self.characters.append(char)
            self.alphabet_size = len(self.characters)

    def add_label(self, label):
        if label not in self.label2index.keys():
            self.label2index[label] = len(self.label2index)
            self.labels.append(label)
            self.number_of_classes = len(self.labels) + 1

    def get_embedding(self, token2vector, embedding_dim):
        unk_vector = np.zeros([embedding_dim])
        return np.array([token2vector.get(token, unk_vector) for token in self.tokens])

    def build_vocabulary(self, tokens ,token_to_vector ):
        for sequence in tokens:
            for token in sequence:
                lower = token.lower()
                if lower in token_to_vector.keys():
                    if lower not in self.token2index.keys():
                        self.token2index[lower] = len(self.tokens)
                        self.tokens.append(lower)
                        self.vocabulary_size = len(self.tokens)

                for char in list(token):
                    self.add_character(char)

    def build_labels(self, labels):
        for label_sequence in labels:
            for label in label_sequence:
                self.add_label(label)

    def transform(self, tokens, labels=None):

        pattern = [[utils_re.get_pattern(token, self.expressions) for token in sequence] for sequence in tokens]
        token_indices = []
        characters = []
        character_indices = []
        token_lengths = []
        character_indices_padded = []
        for token_sequence in tokens:
            token_indices.append(
                [self.token2index.get(token.lower(), self.UNK_INDEX) for token in token_sequence])
            characters.append([list(token) for token in token_sequence])
            character_indices.append(
                [[self.character2index.get(character, 0) for character in token] for token in token_sequence])
            token_lengths.append([len(token) for token in token_sequence])
            longest_token_length_in_sequence = max(token_lengths[-1])
            character_indices_padded.append(
                [utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_INDEX)
                 for temp_token_indices in character_indices[-1]])

        if labels == None:
            return token_indices, character_indices_padded, token_lengths, pattern
        label_indices = []

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
        # self.number_of_classes = len(self.labels) + 1

        if self.verbose:
            print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))
        if self.verbose:
            print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))

        return token_indices, character_indices_padded, token_lengths, pattern, label_indices, label_vector_indices



spacy_nlp = en_core_web_sm.load()

def tokenizer(text):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            # token_dict = {}
            start = token.idx
            end = token.idx + len(token)
            token = text[start:end]
            if token.strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token.split(' ')) != 1:
                print(
                    "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                        token,
                        token.replace(' ', '-')))
                token = token.replace(' ', '-')
            sentence_tokens.append(token)
        sentences.append(sentence_tokens)
    return sentences


# def tokenizer( text):
#     sentences = brat2conll.get_sentences_and_tokens_from_spacy(text, nlp)
#     sentences = [[token['text'] for token in sentence] for sentence in sentences]
#     return sentences

def parse_conll( pathfile):
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
     return tokens , labels


def get_entities_from_brat(text_filepath, annotation_filepath, verbose=False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    if verbose: print("\ntext:\n{0}\n".format(text))

    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = {}
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                if verbose:
                    print("entity: {0}".format(entity))
                # Check compatibility between brat text and anootation
                if utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                        utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                # add to entitys data
                entities.append(entity)
    if verbose: print("\n\n")

    return text, entities

def get_sentences_and_tokens_from_spacy(text):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'] = token.idx
            token_dict['end'] = token.idx + len(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print(
                    "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                        token_dict['text'],
                        token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

def parse_brat(pathfile):
    base_filename = os.path.splitext(os.path.basename(pathfile))[0]
    text_filepath = os.path.join(os.path.dirname(pathfile), base_filename + '.txt')
    annotation_filepath = os.path.join(os.path.dirname(pathfile), base_filename + '.ann')
    # create annotation file if it does not exist
    if not os.path.exists(annotation_filepath):
        codecs.open(annotation_filepath, 'w', 'UTF-8').close()

    text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
    entities = sorted(entities, key=lambda entity: entity["start"])

    # if tokenizer == 'spacy':
    #     sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)
    # elif tokenizer == 'stanford':
    #     sentences = get_sentences_and_tokens_from_stanford(text, core_nlp)

    sentences = get_sentences_and_tokens_from_spacy(text)

    tokens = []
    labels = []
    for sentence in sentences:
        token_sequence = []
        label_sequence = []
        inside = False
        previous_token_label = 'O'
        for token in sentence:
            token['label'] = 'O'
            for entity in entities:
                if entity['start'] <= token['start'] < entity['end'] or \
                                        entity['start'] < token['end'] <= entity['end'] or \
                                                token['start'] < entity['start'] < entity['end'] < token['end']:

                    token['label'] = entity['type'].replace('-',
                                                            '_')  # Because the ANN doesn't support tag with '-' in it

                    break
                elif token['end'] < entity['start']:
                    break

            if len(entities) == 0:
                entity = {'end': 0}
            if token['label'] == 'O':
                gold_label = 'O'
                inside = False
            elif inside and token['label'] == previous_token_label:
                gold_label = 'I-{0}'.format(token['label'])
            else:
                inside = True
                gold_label = 'B-{0}'.format(token['label'])
            if token['end'] == entity['end']:
                inside = False
            previous_token_label = token['label']

            token_sequence.append(token['text'])
            label_sequence.append(gold_label)

            # output_file.write(
            #     '{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'],
            #                                    gold_label))
        tokens.append(token_sequence)
        labels.append(label_sequence)

    return tokens,labels

def parse_brat_folder(path_folder):
    text_filepaths = sorted(glob.glob(os.path.join(path_folder, '*.txt')))
    tokens = []
    labels = []
    for text_file in text_filepaths:
        token , label = parse_brat(text_file)
        tokens.extend(token)
        labels.extend(label)

    return tokens,labels


