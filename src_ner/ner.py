from . import dataset,brat2conll,ner_model,utils,utils_re,utils_nlp




embedding_path = '../../../ML_EntityData/embedding/en/glove.6B.100d.txt'
data_path = {}
data_path['train'] ='../../../ML_EntityData/data/en/train.txt'
data_path['valid'] ='../../../ML_EntityData/data/en/valid.txt'
data_path['test'] = '../../../ML_EntityData/data/en/test.txt'

def main():
    tokens ={}
    labels ={}
    for type in data_path.keys():
        tokens[type],labels[type] = dataset.parse_conll(data_path[type])

    token_to_vector = utils_nlp.load_pretrained_token_embeddings(embedding_path)
    vocab = dataset.Dataset(token_to_vector)
