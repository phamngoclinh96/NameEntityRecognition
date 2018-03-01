import train_P as train
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import utils
import os
import conll2brat
import glob
import codecs
import shutil
import time
import copy
import evaluate
import random
import pickle
import brat2conll
import numpy as np
import utils_nlp
import distutils.util as distutils_util
import configparser
from pprint import pprint
import utils
import utils_data as ds
from BLSTM_CRF_P import BLSTM_CRF



class NER(object):
    def __init__(self,parameters):
        # Initial
        self.prediction_count=0
        # Load dataset
        self.dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters)
        self.dataset = ds.DatasetP(verbose=False, debug=False)
        token_to_vector = self.dataset.load_dataset(self.dataset_filepaths, parameters)

        # Create model lstm+crf
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 2, 'GPU': parameters['number_of_gpus']},
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


    def restore_model_trained(self,parameters,model_pathfile,dataset_pathfile,embedding_filepath,character_dimension,token_dimension):
        self.transition_params_trained = self.model.restore_from_pretrained_model(self.dataset, self.sess,
                                                                        model_pathfile=model_pathfile,
                                                                        dataset_pathfile=dataset_pathfile,
                                                                        embedding_filepath=embedding_filepath,
                                                                        character_dimension=character_dimension,
                                                                        token_dimension=token_dimension,
                                                                        token_to_vector=None)

    def train(self,parameters,number_of_epoch,model_folder):
        stats_graph_folder, experiment_timestamp = utils.create_stats_graph_folder(parameters)

        # Initialize and save execution details
        start_time = time.time()
        # results = {}
        # results['epoch'] = {}
        # results['execution_details'] = {}
        # results['execution_details']['train_start'] = start_time
        # results['execution_details']['time_stamp'] = experiment_timestamp
        # results['execution_details']['early_stop'] = False
        # results['execution_details']['keyboard_interrupt'] = False
        # results['execution_details']['num_epochs'] = 0
        # results['model_options'] = copy.copy(parameters)

        # model_folder = os.path.join(stats_graph_folder, 'model')
        utils.create_folder_if_not_exists(model_folder)

        pickle.dump(self.dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

        bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
        previous_best_valid_f1_score = 0
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
                    transition_params_trained = train.train_step(self.sess, self.dataset, sequence_number, self.model,
                                                                 parameters['dropout_rate'])
                    step += 1
                    if step % 10 == 0:
                        print('Training {0:.2f}% done'.format(step / len(sequence_numbers) * 100), end='\r', flush=True)

            epoch_elapsed_training_time = time.time() - epoch_start_time
            print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)

            y_pred, y_true, output_filepaths = train.predict_labels(sess=self.sess, model=self.model,
                                                                    transition_params_trained=self.transition_params_trained,
                                                                    dataset=self.dataset, epoch_number=epoch_number,
                                                                    stats_graph_folder=stats_graph_folder,
                                                                    dataset_filepaths=self.dataset_filepaths,
                                                                    tagging_format=parameters['tagging_format'])

            #     if epoch_number % 3 ==0:
            self.model.saver.save(self.sess, os.path.join(model_folder, 'model.ckpt'))

            if epoch_number > number_of_epoch:
                break

    def predict(self,text,parameters):
        parameters['dataset_text_folder'] = os.path.join('..', 'data', 'temp')
        stats_graph_folder, _ = utils.create_stats_graph_folder(parameters)

        # Update the deploy folder, file, and dataset
        dataset_type = 'deploy'
        ### Delete all deployment data
        for filepath in glob.glob(os.path.join(parameters['dataset_text_folder'], '{0}*'.format(dataset_type))):
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)
        ### Create brat folder and file
        dataset_brat_deploy_folder = os.path.join(parameters['dataset_text_folder'], dataset_type)
        utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
        dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder, 'temp_{0}.txt'.format(str(self.prediction_count).zfill(5)))  # self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder)
        with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
            f.write(text)
        ### Update deploy filepaths
        dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters,
                                                                                    dataset_types=[dataset_type])
        dataset_filepaths.update(dataset_filepaths)
        dataset_brat_folders.update(dataset_brat_folders)
        ### Update the dataset for the new deploy set
        self.dataset.update_dataset(dataset_filepaths, [dataset_type])

        # Predict labels and output brat
        output_filepaths = {}
        prediction_output = train.prediction_step(self.sess, self.dataset, dataset_type, self.model,
                                                  self.transition_params_trained, stats_graph_folder,
                                                  self.prediction_count, dataset_filepaths, parameters['tagging_format'])
        predictions, _, output_filepaths[dataset_type] = prediction_output

        # print([self.dataset.index_to_label[prediction] for prediction in predictions])
        conll2brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=True)

        # Print and output result
        text_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy',
                                     os.path.basename(dataset_brat_deploy_filepath))
        annotation_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy', '{0}.ann'.format(
            utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
        text2, entities = brat2conll.get_entities_from_brat(text_filepath, annotation_filepath, verbose=True)
        assert (text == text2)
        return entities

    def quick_predict(self,text,folder='../deploy'):
        dataset_type = self.dataset.create_deploy_set(text)

        # Predict labels and output brat
        # output_filepaths = {}
        prediction_output = train.prediction(self.sess, self.dataset, dataset_type, self.model,
                                                  self.transition_params_trained)
        # predictions, _, output_filepaths[dataset_type] = prediction_output
        print(prediction_output)
        for i,sentence in enumerate(prediction_output):
            token = ''
            for j,label in enumerate(sentence):
                if label!= 'O':
                    label = label.split('-')
                    prefix = label[0]
                    entity = label[1]


        return prediction_output
        # print([self.dataset.index_to_label[prediction] for prediction in predictions])