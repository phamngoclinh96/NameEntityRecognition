import ner



path_model = '../model/en'

name_entity = ner.restore_model_trained(model_folder=path_model,
                                        embedding_filepath='../../../ML_EntityData/embedding/en/glove.6B.100d.txt')

print('Loaded model.')
print('predict')
while True:
    text=input('Input :')
    if text == 'exit':
        break
    _,entitys = name_entity.quick_predict(text)
    for entity in entitys:
        print(entity)