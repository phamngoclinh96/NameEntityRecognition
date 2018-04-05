from name_entity import dataset,ner_model,utils_nlp

import tensorflow as tf
import pickle
import os
import codecs

model_path = '../../model/threatintelligence'

def predict(text,vocab,model,sess):
    # print(text)
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

def get_entities(sentences,labels):
    tokens = []
    entitys = []
    for i, sentence in enumerate(labels):
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

    vocab = pickle.load(open(os.path.join(model_path, 'vocab.pickle'),'rb'))
    sess = tf.Session()
    model= ner_model.BLSTM_CRF(vocab,token_embedding_dimension=100,character_lstm_hidden_state_dimension=50,token_lstm_hidden_state_dimension=50,character_embedding_dimension=25)
    sess.run(tf.global_variables_initializer())
    model.load_model(sess, os.path.join(model_path, 'model.ckpt'))
    filename = '0002.txt'
    text = codecs.open('../data/test/'+filename,'r','utf-8').read()
    entities = predict(text,vocab,model,sess)
    output = codecs.open('../data/output/'+filename,'w','utf-8')
    for entity,name in entities:
        print(entity,':',name)
        output.writelines(entity+':'+name+'\n\r')

    output.close()

main()
