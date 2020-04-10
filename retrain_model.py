import tensorflow.keras.models import load_models
import tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.optimizers import Adam
import pandas as pd



def retrain(model,input_shape,training_set,batch_size=64,learning_rate):
    def load_data():
        dataframe = pd.read_csv(trainin_set)
        ##  load data and nomralize it ...
        ##   TODO
    model = load_model(model)
    import time
    retrain_time = time.time()
    retrain_time = int(retrain_time)
    print("Retrain time is {}".format(retrain_time)) 
    optimizer = Adam(lr=learning_rate)
    checkpoint = ModelCheckpoint(filepath="checkpoint_{}.h5".format(retrain_time),save_best_only=True,monitor="val_loss")
    hist = model.fit()
    pass

if  __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",requied=True)
    ap.add_argument("--input-shape",required=True,type=int)
    ap.add_argument("-p","--training-set-path",required=True)
    args = vars(ap.parser_args())
    model = args["model"]
    input_shape= args["input_shape"]
    training_set = args["trainin_set_path"]
    retrain(model,input_shape,training_set)
    print("Successfully retrain model")
