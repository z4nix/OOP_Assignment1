import json
import pickle

class ModelSaver:
    def __init__(self, format='json') -> None:
        if format not in ['json', 'pickle']:
            raise ValueError("Unsupported format. Please use 'json' or 'pickle'.") #exception to catch unsupported format (ValueError)
        self.format = format
    
    def save_model(self, model, filename) -> None:
        params = model.get_params()
        try:
            if self.format == 'json':
                with open(filename, 'w') as file:
                    json.dump(params, file)
            elif self.format == 'pickle':
                with open(filename, 'wb') as file:
                    pickle.dump(params, file)
        except Exception as e:
            print(f"an error occurred while saving the model: {e}") #Saving unsuccesful

    def load_model(self, filename, model) -> None:
        try:
            if self.format == 'json':
                with open(filename, 'r') as file:
                    params = json.load(file)
            elif self.format == 'pickle':
                with open(filename, 'rb') as file:
                    params = pickle.load(file)
            model.set_params(params)
        except FileNotFoundError:
            print(f"File {filename} not found.") #handles files that have not been found
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")


