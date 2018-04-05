from name_entity import dataset,ner_model,utils_nlp

import tensorflow as tf
import pickle
import numpy as np
import os

embedding_path = '../../../ML_EntityData/embedding/en/glove.6B.100d.txt'
data_path = {}
# data_path['train'] ='../../../ML_EntityData/data/en/train.txt'
# data_path['valid'] ='../../../ML_EntityData/data/en/valid.txt'
# data_path['test'] = '../../../ML_EntityData/data/en/test.txt'

data_path['train'] ='../../../ML_EntityData/data/report/train'
data_path['valid'] ='../../../ML_EntityData/data/report/valid'
data_path['news'] = '../../../ML_EntityData/data/report/news'
data_path['news2'] = '../../../ML_EntityData/data/news'
model_path = '../../model/threatintelligence'


def main():
    tokens ={}
    labels ={}
    print('Load dataset')
    for type in data_path.keys():
        tokens[type],labels[type] = dataset.parse_brat_folder(data_path[type])
        labels[type] = [utils_nlp.bio_to_bioes(label) for label in labels[type]]

    vocab = pickle.load(open(os.path.join(model_path, 'vocab.pickle'), 'rb'))
    # print('Load embedding ..')
    # token_to_vector = utils_nlp.load_pretrained_token_embeddings(embedding_path)
    # vocab = dataset.Dataset()
    #
    #
    # print('Build vocabulary')
    # for type in data_path.keys():
    #     vocab.build_vocabulary(tokens[type], token_to_vector)
    #     vocab.build_labels(labels[type])
    #
    # pickle.dump(vocab, open(os.path.join(model_path, 'vocab.pickle'), 'wb'))


    print('Convert to indices')

    token_indices = {}
    character_indices_padded ={}
    token_lengths = {}
    pattern ={}
    label_indices ={}
    label_vector_indices = {}
    for type in data_path.keys():
        token_indices[type],character_indices_padded[type],token_lengths[type],pattern[type],label_indices[type],label_vector_indices[type] = vocab.transform(tokens[type],labels[type])

    print('Create model')
    sess = tf.Session()
    model= ner_model.BLSTM_CRF(vocab,token_embedding_dimension=100,character_lstm_hidden_state_dimension=50,token_lstm_hidden_state_dimension=50,character_embedding_dimension=25)
    sess.run(tf.global_variables_initializer())
    model.load_model(sess, os.path.join(model_path, 'model.ckpt'))

    print('Evaluate...')
    # model.load_token_embedding(sess,vocab,token_to_vector,100)


    for datatype in data_path.keys():
        print('evaluate',datatype)
        model.evaluate(sess,vocab,token_indices[datatype],character_indices_padded[datatype],token_lengths[datatype],pattern[datatype],label_indices[datatype],datatype)



    # print(predict("McLellan says that his group is trying to raise awareness of the problem so that companies will see cryptocurrency miners as a security issue on the same level as banking Trojans and other well-known types of malware because monitoring networks are seeing a shift to the miners from older types of intrusion. \"I think a lot of organizations will have these on their networks,\" he says, simply because they're becoming a popular way for criminals to make money.",vocab,model,sess))

main()
