



class Vocabulary(object):
    def __init__(self):
        self.token2index ={}
        self.tokens = []
        self.characters = []
        self.character2index = {}

        self.UNK = 'UNK'
        self.index_token[self.UNK] = 0
