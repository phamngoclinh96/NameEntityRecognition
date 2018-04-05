import tensorflow as tf
import pickle
import ner
import ner_model

def main_test():
    vocab = pickle.load(open('../../model/vocab.pickle','rb'))

    sess = tf.Session()
    model= ner_model.BLSTM_CRF(vocab,token_embedding_dimension=100,character_lstm_hidden_state_dimension=50,token_lstm_hidden_state_dimension=50,character_embedding_dimension=25)
    sess.run(tf.global_variables_initializer())
    # model.load_token_embedding(sess,vocab,token_to_vector,100)
    # sess.run(model.token_embedding_weights.assign(np.ndarray([vocab.vocabulary_size,100])))


    model.load_model(sess,'../../model/model.ckpt')

    print(ner.predict('my name is Ngoc Linh',vocab,model,sess))

main_test()