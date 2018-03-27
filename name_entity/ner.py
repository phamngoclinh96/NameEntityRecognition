import dataset,ner_model,utils_nlp

import tensorflow as tf
import pickle
import numpy as np

embedding_path = '../../../ML_EntityData/embedding/en/glove.6B.100d.txt'
data_path = {}
data_path['train'] ='../../../ML_EntityData/data/en/train.txt'
data_path['valid'] ='../../../ML_EntityData/data/en/valid.txt'
data_path['test'] = '../../../ML_EntityData/data/en/test.txt'


def predict(text,vocab,model,sess):
    print(text)
    text = text + '.'
    sentences = dataset.tokenizer(text)
    token_indices_test, character_indices_padded_test, token_lengths_test, pattern_test = vocab.transform(sentences)
    pre = model.predict(sess, token_indices_test, character_indices_padded_test, token_lengths_test, pattern_test)

    prediction_output = [[vocab.labels[i] for i in sentence] for sentence in pre]

    tokens = []
    entitys = []
    for i, sentence in enumerate(prediction_output):
        token = ''
        previous_label = 'O'
        sentence = utils_nlp.bioes_to_bio(sentence)
        for j, label in enumerate(sentence):
            if label != 'O':
                label = label.split('-')
                prefix = label[0]
                if prefix == 'B' or previous_label != label[1]:
                    if previous_label != 'O':
                        tokens.append(token)
                        entitys.append(previous_label)
                    previous_label = label[1]
                    token = sentences[i][
                        j]  # self.dataset.index_to_token[self.dataset.token_indices[dataset_type][i][j]]
                else:
                    token = token + ' ' + sentences[i][
                        j]  # self.dataset.index_to_token[self.dataset.token_indices[dataset_type][i][j]]

            else:
                if previous_label != 'O':
                    tokens.append(token)
                    entitys.append(previous_label)
                    token = ''
                previous_label = 'O'
    return list(zip(tokens, entitys))

def main():
    tokens ={}
    labels ={}
    print('Load embedding ..')
    # token_to_vector = utils_nlp.load_pretrained_token_embeddings(embedding_path)
    # vocab = dataset.Dataset()
    vocab = pickle.load(open('../../model/vocab.pickle','rb'))
    for type in data_path.keys():
        tokens[type],labels[type] = dataset.parse_conll(data_path[type])
        labels[type] = [utils_nlp.bio_to_bioes(label) for label in labels[type]]
        # vocab.build_vocabulary(tokens[type], token_to_vector)
        # vocab.build_labels(labels[type])

    # pickle.dump(vocab,open('../../model/vocab.pickle','wb'))
    token_indices = {}
    character_indices_padded ={}
    token_lengths = {}
    pattern ={}
    label_indices ={}
    label_vector_indices = {}
    for type in data_path.keys():
        token_indices[type],character_indices_padded[type],token_lengths[type],pattern[type],label_indices[type],label_vector_indices[type] = vocab.transform(tokens[type],labels[type])

    sess = tf.Session()
    model= ner_model.BLSTM_CRF(vocab,token_embedding_dimension=100,character_lstm_hidden_state_dimension=50,token_lstm_hidden_state_dimension=50,character_embedding_dimension=25)
    sess.run(tf.global_variables_initializer())
    model.load_model(sess,'../../model/model.ckpt')
    # model.load_token_embedding(sess,vocab,token_to_vector,100)

    for epoch in range(10):
        type= 'train'
        model.train_step(sess,token_indices[type],character_indices_padded[type],token_lengths[type],pattern[type],label_indices[type],label_vector_indices[type],0.5)

        model.saver.save(sess,'../../model/model.ckpt')

    # model.load_model(sess,'../../model/model.ckpt')

    print(predict('my name is Ngoc Linh',vocab,model,sess))

main()
