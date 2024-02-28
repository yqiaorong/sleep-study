import pickle

# load the model 
model_path = 'dataset/THINGS_EEG2/reg_model.pkl'
reg = pickle.load(open(model_path, 'rb'))