import json
import pickle

class ModelSaver:
    def __init__(self, format='json') -> None:
        self.format = format
    
    def save_model(self, model, filename) -> None:
        params = model.get_params()
        if self.format == 'json':
            with open(filename, 'w') as file: # create file in write mode
                json.dump(params, file)
        elif self.format == 'pickle':
            with open(filename, 'wb') as file: # create file in write binary mode
                pickle.dump(params, file)

    def load_model(self, filename, model) -> None:
        if self.format == 'json':
            with open(filename, 'r') as file: # open file with read mode
                params = json.load(file)
        elif self.format == 'pickle':
            with open(filename, 'rb') as file: # open file with read binary mode
                params = pickle.load(file)
        model.set_params(params)
