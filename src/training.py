import gc
import numpy as np
from tensorflow import keras
import os
import pandas as pd
import sys

os.chdir(os.path.dirname(__file__))
os.chdir("../project_files")
sys.path.append(os.getcwd())
import golois





def init():
    N = 10000
    planes = 31
    moves = 361
    filters = 32

    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype ('float32')

    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical (policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype ('float32')

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype ('float32')

    groups = np.zeros((N, 19, 19, 1))
    groups = groups.astype ('float32')

    golois.getValidation (input_data, policy, value, end)
    return input_data, policy, value, end, groups


def reloading(name):
    try : 
        return name + ".h5" in os.listdir("../save")
    except:
        print(os.listdir("../save"))
        print(name + ".h5")
        return False

def reload(name):
    model = keras.models.load_model("../save/" + name + ".h5")
    hist_T = pd.read_csv("../save/"+name + "_training.csv")
    hist_V = pd.read_csv("../save/"+name + "_validation.csv")
    return model, hist_T, hist_V

def train(model, epochs, name, batch = 128):
    print("train")
    N = 10000
    input_data, policy, value, end, groups = init()
    if reloading(name):
        print("Reloading")
        model, hist_T, hist_V = reload(name)
        print(len(hist_T))
        
        start = len(hist_T) + 1
    else:
        print("Not reloading")
        start = 1
        hist_T = pd.DataFrame(columns = ["loss","policy_loss", "value_loss", "policy_categorical_accuracy", "value_mse"])
        hist_V = pd.DataFrame(columns=["loss","policy_loss", "value_loss", "policy_categorical_accuracy", "value_mse"])
    for i in range (start, epochs + 1):
        print ('epoch ' + str (i))
        golois.getBatch (input_data, policy, value, end, groups, i * N)
        history = model.fit(input_data,
                            {'policy': policy, 'value': value}, 
                            epochs=1, batch_size=batch)
        h = history.history
        h['loss'] = h["loss"][0]
        h['policy_loss'] = h["policy_loss"][0]
        h['value_loss'] = h["value_loss"][0]
        h['policy_categorical_accuracy'] = h["policy_categorical_accuracy"][0]
        h['value_mse'] = h["value_mse"][0]
        hist_T.loc[len(hist_T)] = h
        if (i % 5 == 0):
            gc.collect ()
        if (i % 20 == 0):
            golois.getValidation (input_data, policy, value, end)
            val = model.evaluate (input_data,
                                  [policy, value], verbose = 0, batch_size=batch)
            hist_V = hist_V.append(pd.Series(val, index=hist_V.columns), ignore_index=True)
            model.save ('../save/'+name+'.h5')
            hist_T.to_csv("../save/"+ name +"_training.csv" , index =False)
            hist_V.to_csv("../save/"+ name +"_validation.csv", index = False)
        if i == 2001:
            keras.backend.set_value(model.optimizer.learning_rate, 0.005)
        if i == 3501:
            keras.backend.set_value(model.optimizer.learning_rate, 0.0005)
        if i == 4501:
            keras.backend.set_value(model.optimizer.learning_rate, 0.00005)