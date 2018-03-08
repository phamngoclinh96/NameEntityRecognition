import en_core_web_sm

import train
import tensorflow as tf

import os
import conll2brat
import glob
import codecs
import shutil
import time

import random
import pickle
import brat2conll
import numpy as np
import utils_nlp
import utils
import utils_data as ds
from ner_model import BLSTM_CRF


def restore_model_trained(parameter_pathfile='',model_pathfile='',dataset_pathfile='',embedding_filepath='',model_folder = ''):
    if model_folder=='':
        parameters = pickle.load(open(parameter_pathfile,'rb'))
        token_to_vector = utils_nlp.load_pretrained_token_embeddings(embedding_filepath)
        dataset= pickle.load(open(dataset_pathfile,'rb'))
        name_entity = NER(parameters,dataset,token_to_vector)
        name_entity.restore_model_trained(model_pathfile,dataset_pathfile,embedding_filepath,character_dimension=parameters['character_embedding_dimension'],
                                          token_dimension=parameters['token_embedding_dimension'])
    else:
        parameter_pathfile=os.path.join(model_folder,'parameters.pickle')
        dataset_pathfile =os.path.join(model_folder,'dataset.pickle')
        model_pathfile = os.path.join(model_folder,'model.ckpt')
        parameters = pickle.load(open(parameter_pathfile, 'rb'))
        token_to_vector = utils_nlp.load_pretrained_token_embeddings(embedding_filepath)
        dataset = pickle.load(open(dataset_pathfile, 'rb'))
        name_entity = NER(parameters, dataset, token_to_vector)
        name_entity.restore_model_trained(model_pathfile, dataset_pathfile, embedding_filepath,
                                          character_dimension=parameters['character_embedding_dimension'],
                                          token_dimension=parameters['token_embedding_dimension'])

    return name_entity


class NER(object):
    def __init__(self,parameters,dataset = None,token_to_vector=None,tagging_format = 'bioes',number_of_cpu = 1,number_of_cpu_threads=4,number_of_gpus=0):
        # Initial
        self.prediction_count=0
        self.nlp = en_core_web_sm.load()
        self.tagging_format =tagging_format
        self.parameters = parameters
        # Load dataset
        if dataset == None:
            self.dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters['dataset_text_folder'])
            self.dataset = ds.Dataset(verbose=False, debug=False)
            token_to_vector = self.dataset.load_dataset(self.dataset_filepaths, parameters['token_pretrained_embedding_filepath'], parameters,load_all_pretrained_token_embeddings = False ,tagging_format=tagging_format)
        else:
            self.dataset = dataset

        # Create model lstm+crf
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=number_of_cpu_threads,
            inter_op_parallelism_threads=number_of_cpu_threads,
            device_count={'CPU': number_of_cpu, 'GPU': number_of_gpus},
            allow_soft_placement=True,
            # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
            log_device_placement=False
        )
        self.sess = tf.Session(config=session_conf)

        with self.sess.as_default():
            # Create model and initialize or load pretrained model
            ### Instantiate the model
            self.model = BLSTM_CRF(self.dataset, token_embedding_dimension=parameters['token_embedding_dimension'],
                              character_lstm_hidden_state_dimension=parameters['character_lstm_hidden_state_dimension'],
                              token_lstm_hidden_state_dimension=parameters['token_lstm_hidden_state_dimension'],
                              character_embedding_dimension=parameters['character_embedding_dimension'],
                              gradient_clipping_value=parameters['gradient_clipping_value'],
                              learning_rate=parameters['learning_rate'],
                              freeze_token_embeddings=parameters['freeze_token_embeddings'],
                              optimizer=parameters['optimizer'],
                              maximum_number_of_epochs=parameters['maximum_number_of_epochs'])

        self.sess.run(tf.global_variables_initializer())

        self.model.load_pretrained_token_embeddings(self.sess, self.dataset,
                                               embedding_filepath=parameters['token_pretrained_embedding_filepath'],
                                               check_lowercase=parameters['check_for_lowercase'],
                                               check_digits=parameters['check_for_digits_replaced_with_zeros'],
                                               token_to_vector=token_to_vector)
        # Initial params_train
        self.transition_params_trained = np.random.rand(len(self.dataset.unique_labels) + 2, len(self.dataset.unique_labels) + 2)


    def restore_model_trained(self,model_pathfile,dataset_pathfile,embedding_filepath,character_dimension,token_dimension):
        self.transition_params_trained = self.model.restore_from_pretrained_model(self.dataset, self.sess,
                                                                        model_pathfile=model_pathfile,
                                                                        dataset_pathfile=dataset_pathfile,
                                                                        embedding_filepath=embedding_filepath,
                                                                        character_dimension=character_dimension,
                                                                        token_dimension=token_dimension,
                                                                        token_to_vector=None)

    def save_to(self,model_folder):

        pickle.dump(self.dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))
        pickle.dump(self.parameters, open(os.path.join(model_folder, 'parameters.pickle'), 'wb'))
        self.model.saver.save(self.sess, os.path.join(model_folder, 'model.ckpt'))
        return model_folder


    def train(self,max_number_of_epoch,model_folder,dropout_rate=0.5):
        # stats_graph_folder, experiment_timestamp = utils.create_stats_graph_folder(parameters)

        # Initialize and save execution details
        start_time = time.time()

        utils.create_folder_if_not_exists(model_folder)

        pickle.dump(self.dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))
        pickle.dump(self.parameters,open(os.path.join(model_folder,'parameters.pickle'),'wb'))

        bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
        previous_best_valid_f1_score = -100
        epoch_number = -1

        while True:

            step = 0
            epoch_number += 1
            print('\nStarting epoch {0}'.format(epoch_number))

            epoch_start_time = time.time()

            if epoch_number != 0:
                # Train model: loop over all sequences of training set with shuffling
                sequence_numbers = list(range(len(self.dataset.token_indices['train'])))
                random.shuffle(sequence_numbers)
                for sequence_number in sequence_numbers:
                    self.transition_params_trained = train.train_step(self.sess, self.dataset, sequence_number, self.model,
                                                                 dropout_rate)
                    step += 1
                    if step % 10 == 0:
                        print('Training {0:.2f}% done'.format(step / len(sequence_numbers) * 100), end='\r', flush=True)

            epoch_elapsed_training_time = time.time() - epoch_start_time
            print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)
            f1_score ={}
            for data_type in ['train', 'valid', 'test']:
                if data_type not in self.dataset.label_indices.keys():
                    continue
                _,_,f1_score[data_type] = train.evaluate_step(sess=self.sess,dataset_type=data_type, dataset=self.dataset, model=self.model,
                                transition_params_trained=self.transition_params_trained,
                                tagging_format=self.tagging_format)
            #     if epoch_number % 3 ==0:
            self.model.saver.save(self.sess, os.path.join(model_folder, 'model.ckpt'))
            if abs(f1_score['valid'][-2] - previous_best_valid_f1_score) < 0.1:
                bad_counter+=1
            if bad_counter>10:
                break
            previous_best_valid_f1_score =f1_score['valid'][-2]
            if epoch_number > max_number_of_epoch:
                break

    # def predict(self,text,parameters):
    #     parameters['dataset_text_folder'] = os.path.join('..', 'data', 'temp')
    #     stats_graph_folder, _ = utils.create_stats_graph_folder(parameters)
    #
    #     # Update the deploy folder, file, and dataset
    #     dataset_type = 'deploy'
    #     ### Delete all deployment data
    #     for filepath in glob.glob(os.path.join(parameters['dataset_text_folder'], '{0}*'.format(dataset_type))):
    #         if os.path.isdir(filepath):
    #             shutil.rmtree(filepath)
    #         else:
    #             os.remove(filepath)
    #     ### Create brat folder and file
    #     dataset_brat_deploy_folder = os.path.join(parameters['dataset_text_folder'], dataset_type)
    #     utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
    #     dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder, 'temp_{0}.txt'.format(str(self.prediction_count).zfill(5)))  # self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder)
    #     with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
    #         f.write(text)
    #     ### Update deploy filepaths
    #     dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters,
    #                                                                                 dataset_types=[dataset_type])
    #     dataset_filepaths.update(dataset_filepaths)
    #     dataset_brat_folders.update(dataset_brat_folders)
    #     ### Update the dataset for the new deploy set
    #     self.dataset.update_dataset(dataset_filepaths, [dataset_type])
    #
    #     # Predict labels and output brat
    #     output_filepaths = {}
    #     prediction_output = train.prediction_step(self.sess, self.dataset, dataset_type, self.model,
    #                                               self.transition_params_trained, stats_graph_folder,
    #                                               self.prediction_count, dataset_filepaths, parameters['tagging_format'])
    #     predictions, _, output_filepaths[dataset_type] = prediction_output
    #
    #     # print([self.dataset.index_to_label[prediction] for prediction in predictions])
    #     conll2brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=True)
    #
    #     # Print and output result
    #     text_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy',
    #                                  os.path.basename(dataset_brat_deploy_filepath))
    #     annotation_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy', '{0}.ann'.format(
    #         utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
    #     text2, entities = brat2conll.get_entities_from_brat(text_filepath, annotation_filepath, verbose=True)
    #     assert (text == text2)
    #     return entities

    def quick_predict(self,text):
        text = text+'.'
        sentences = brat2conll.get_sentences_and_tokens_from_spacy(text, self.nlp)
        dataset_type = self.dataset.create_deploy_set(sentences)

        # Predict labels and output brat
        # output_filepaths = {}
        prediction_output = train.prediction(self.sess, self.dataset, dataset_type, self.model,
                                                  self.transition_params_trained)
        # predictions, _, output_filepaths[dataset_type] = prediction_output
        # print(prediction_output)
        tokens=[]
        entitys=[]
        for i,sentence in enumerate(prediction_output):
            token = ''
            previous_label= 'O'
            sentence = utils_nlp.bioes_to_bio(sentence)
            for j,label in enumerate(sentence):
                if label!= 'O':
                    label = label.split('-')
                    prefix = label[0]
                    if prefix == 'B' or previous_label != label[1]:
                        if previous_label != 'O':
                            tokens.append(token)
                            entitys.append(previous_label)
                            token = ''
                        previous_label = label[1]
                        token = sentences[i][j]['text']  #self.dataset.index_to_token[self.dataset.token_indices[dataset_type][i][j]]
                    else:
                        token = token + ' ' + sentences[i][j]['text'] # self.dataset.index_to_token[self.dataset.token_indices[dataset_type][i][j]]

                else:
                    if previous_label != 'O':
                        tokens.append(token)
                        entitys.append(previous_label)
                        token=''
                    previous_label = 'O'

        return prediction_output , list(zip(tokens,entitys))
        # print([self.dataset.index_to_label[prediction] for prediction in predictions])