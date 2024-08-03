from datetime import datetime
import os
import random
import sys

from .model import SSCModel as CustomModel
from .utils import json_processor


from keras_preprocessing import sequence
import matplotlib.pyplot as plt
import nltk
import numpy
from sklearn.utils import class_weight
import tensorflow as tf




class LstmTrainer:
    def __init__(self, args):
        self.model = CustomModel(args)
        self.model.load_w2v_model()
        self.w2v_model = self.model.w2v_model
        self.word_vectors = self.model.word_vectors     
        self.w2v_model_save_dir = args.w2v_model_save_dir
        self.w2v_mincount = args.w2v_mincount
        self.w2v_epochs = args.w2v_epochs
        self.w2v_embedding_dim = args.w2v_embedding_dim
        self.input_dir_good = args.input_dir_good
        self.input_dir_bad = args.input_dir_bad
        self.json_types = args.json_types
        self.train_ratio = args.train_ratio
        self.test_ratio = args.test_ratio
        self.val_ratio = args.val_ratio
        self.dropout = args.dropout
        self.neurons = args.neurons
        self.optimizer = args.optimizer
        self.epochs = args.epochs
        self.batchsize = args.batchsize
        self.enable_class_weights = args.enable_class_weights
        self.lstm_model_save_dir = args.lstm_model_save_dir
        self.model_type = args.model
        self.w2v_model_name = args.w2v_model_name

        self.complete_structure = args.need_complete_structure
        self.kv_type = args.kv_type

        self.max_len = 50
        self.models = self.model.models

    def shuffle_data(self, data):
        keys = []

        #randomize the sample and split into train, validate and final test set
        for i in range(len(data)):
            keys.append(i)
            random.shuffle(keys)
        return keys

    def cutoff_data(self, data):
        keys = self.shuffle_data(data)
        cutoff = round(self.train_ratio * len(keys)) #     train_ratio% for the training set
        cutoff2 = round((self.train_ratio + self.test_ratio) * len(keys)) #   finaltest_ratio% for the validation set and test_ratio% for the final test set

        keystrain = keys[:cutoff]
        keystest = keys[cutoff:cutoff2]
        keysfinaltest = keys[cutoff2:]

        return keystrain, keystest, keysfinaltest

    def data_generator(self, keys, datas):
        valid_words = set(self.word_vectors.key_to_index)
        debug_set = []
        for k in keys:
            block = datas[k]    
            X, y = block
            token = (X)
            
            if not token:
                continue

            if len(token) > 1000:
                for i in range(0, len(token), 1000):
                    vectorlist = [
                        self.w2v_model.wv[word].tolist() 
                        for t in token[i:i+1000] 
                        if t and t != " " 
                        for word in nltk.word_tokenize(str(t)) 
                        if word in valid_words and word != " "
                    ]
                    yield numpy.array(vectorlist), y, debug_set

            else:
                vectorlist = [
                    self.w2v_model.wv[word].tolist() 
                    for t in token 
                    if t and t != " " 
                    for word in nltk.word_tokenize(str(t)) 
                    if word in valid_words and word != " "
                ]
                yield numpy.array(vectorlist), y, debug_set

            

            

    def create_dataset(self, keys, datas, Xset, Yset):
        dir_path = os.path.join(os.getcwd(), 'data', 'npy', 'temp')
        os.makedirs(dir_path, exist_ok=True)

        generator = self.data_generator(keys, datas)
        idx = 0
        for batch_X, batch_Y, _ in generator:
            Xset.extend([batch_X])
            Yset.extend([batch_Y])

            if len(Xset) >= 1000:
                Xset = sequence.pad_sequences(Xset, padding='post', dtype='float32', maxlen=self.max_len)
                numpy.save(os.path.join(dir_path, f'Xset_{idx}'), Xset)
                numpy.save(os.path.join(dir_path, f'Yset_{idx}'), numpy.array(Yset))
                Xset = []
                Yset = []
                idx += 1
        
        if len(Xset) > 0:
            Xset = sequence.pad_sequences(Xset, padding='post', dtype='float32', maxlen=self.max_len)
            numpy.save(os.path.join(dir_path, f'Xset_{idx}'), Xset)
            numpy.save(os.path.join(dir_path, f'Yset_{idx}'), numpy.array(Yset))
            Xset = []
            Yset = []
            idx += 1

        for i in range(idx):
            Xset.extend(numpy.load(os.path.join(dir_path, f'Xset_{i}.npy')))
            Yset.extend(numpy.load(os.path.join(dir_path, f'Yset_{i}.npy')))
        
        return Xset, Yset
    
    def read_data(self):
        json_types = self.json_types
        goodblocks = []
        for folder_name in self.input_dir_good:
            dir_path = os.path.join(os.getcwd(), folder_name)
            goodblocks.extend(json_processor.traverse_and_read_json_with_label(dir_path, json_types, self.kv_type, 0, self.complete_structure))

        badblocks = []
        for folder_name in self.input_dir_bad:
            dir_path = os.path.join(os.getcwd(), folder_name)
            badblocks.extend(json_processor.traverse_and_read_json_with_label(dir_path, json_types, self.kv_type, 1, self.complete_structure))

        allblocks = []
        allblocks.extend(badblocks)
        allblocks.extend(goodblocks)
        print("len of allblock is:", len(allblocks))
        memory_size = sys.getsizeof(allblocks)
        print(f"Memory size of allblocks: {memory_size} bytes")


        keystrain, keystest, keysfinaltest = self.cutoff_data(allblocks)
        TrainX = []
        TrainY = []
        ValidateX = []
        ValidateY = []
        FinaltestX = []
        FinaltestY = []
        print("Creating training dataset...")
        TrainX, TrainY = self.create_dataset(keystrain, allblocks, TrainX, TrainY)

        print("Creating validation dataset...")
        ValidateX, ValidateY = self.create_dataset(keystest, allblocks, ValidateX, ValidateY)

        print("Creating finaltest dataset...")
        FinaltestX, FinaltestY = self.create_dataset(keysfinaltest, allblocks, FinaltestX, FinaltestY)

        print("Train length: " + str(len(TrainX)))
        print("Test length: " + str(len(ValidateX)))
        print("Finaltesting length: " + str(len(FinaltestX)))
        now = datetime.now() # current date and time
        nowformat = now.strftime("%H:%M")
        print("time: ", nowformat)

        
        #padding sequences on the same length
        TrainX = sequence.pad_sequences(TrainX, padding='post', dtype='float32', maxlen=self.max_len)
        ValidateX = sequence.pad_sequences(ValidateX, padding='post', dtype='float32', maxlen=self.max_len)
        FinaltestX = sequence.pad_sequences(FinaltestX, padding='post', dtype='float32', maxlen=self.max_len)

        return TrainX, TrainY, ValidateX, ValidateY, FinaltestX, FinaltestY
    
    def get_numpy_data(self):

        dir_path = os.path.join(os.getcwd(), 'data', 'npy')
        os.makedirs(dir_path, exist_ok=True)

        X_train_path = os.path.join(dir_path, f'{self.json_types}-X_train.npy')
        y_train_path = os.path.join(dir_path, f'{self.json_types}-y_train.npy')
        X_test_path = os.path.join(dir_path, f'{self.json_types}-X_test.npy')
        y_test_path = os.path.join(dir_path, f'{self.json_types}-y_test.npy')
        X_finaltest_path = os.path.join(dir_path, f'{self.json_types}-X_finaltest.npy')
        y_finaltest_path = os.path.join(dir_path, f'{self.json_types}-y_finaltest.npy')

        if os.path.exists(X_train_path):
            X_train =  numpy.load(X_train_path)
            y_train =  numpy.load(y_train_path)
            X_test =  numpy.load(X_test_path)
            y_test =  numpy.load(y_test_path)
            X_finaltest =  numpy.load(X_finaltest_path)
            y_finaltest =  numpy.load(y_finaltest_path)
            return X_train, y_train, X_test, y_test, X_finaltest, y_finaltest

        TrainX, TrainY, ValidateX, ValidateY, FinaltestX, FinaltestY = self.read_data()

        X_train =  numpy.array(TrainX)
        y_train =  numpy.array(TrainY)
        X_test =  numpy.array(ValidateX)
        y_test =  numpy.array(ValidateY)
        X_finaltest =  numpy.array(FinaltestX)
        y_finaltest =  numpy.array(FinaltestY)

        numpy.save(X_train_path, X_train)
        numpy.save(y_train_path, y_train)
        numpy.save(X_test_path, X_test)
        numpy.save(y_test_path, y_test)
        numpy.save(X_finaltest_path, X_finaltest)
        numpy.save(y_finaltest_path, y_finaltest)

        return X_train, y_train, X_test, y_test, X_finaltest, y_finaltest


    def fit(self):
        X_train, y_train, X_test, y_test, X_finaltest, y_finaltest = self.get_numpy_data()

        print("numpy array done. ", datetime.now().strftime("%H:%M"))

        print(str(len(X_train)) + " samples in the training set.")      
        print(str(len(X_test)) + " samples in the validation set.") 
        print(str(len(X_finaltest)) + " samples in the final test set.")
        
        csum = 0
        for a in y_train:
            csum = csum+a
        print("percentage of vulnerable samples: "  + str(int((csum / len(X_train)) * 10000)/100) + "%")
        
        testvul = 0
        for y in y_test:
            if y == 1:
                testvul = testvul+1
        print("absolute amount of vulnerable samples in test set: " + str(testvul))

        epochs = self.epochs
        batchsize = self.batchsize

        print("Starting Model: ", datetime.now().strftime("%H:%M"))
        f1_loss, f1 = self.model.get_evaluate_func()
        
        model = self.models[self.model_type]
        model.compile(loss=f1_loss, optimizer=self.optimizer, metrics=f1)
        print("Compiled Model: ", datetime.now().strftime("%H:%M"))

        os.makedirs('./pic', exist_ok=True)
        print("X_train.shape = ", X_train.shape)
        print("y_train.shape = ", y_train.shape)
        model.build(input_shape=X_train.shape)
        model.summary()
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_test, y_test)) #epochs more are good, batch_size more is good
        plt = self.plot_history(history)
        plt.savefig(f'./pic/{self.model_type}-{self.json_types}-plot-history.png')

        result = {}
        datasets = {
            "train": (X_train, y_train),
            "validation": (X_test, y_test),
            "test": (X_finaltest, y_finaltest)
        }

        os.makedirs('./numpy', exist_ok=True)
        for dataset, (X_set, y_set) in datasets.items():
            print("Now predicting on " + dataset + " set (" + str(self.dropout) + " dropout)")

            accuracy, precision, recall, F1Score, fpr, tpr = self.model.evaluate(model, X_set, y_set)
            
            numpy.save(f'./{self.model_type}-{self.json_types}-{dataset}-tpr.npy', tpr)
            numpy.save(f'./{self.model_type}-{self.json_types}-{dataset}-fpr.npy', fpr)
            plt = self.plot_roc(fpr, tpr)
            plt.savefig(f'./pic/{self.model_type}-{self.json_types}-{dataset}-plot-roc.png')


            result[dataset] = [str(accuracy), str(precision), str(recall), str(F1Score)]
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print('F1 score: ', F1Score, "\n")

        print("saving model. ", datetime.now().strftime("%H:%M"))
        os.makedirs(self.lstm_model_save_dir, exist_ok=True)
        model.save_weights(self.lstm_model_save_dir + f'{self.model_type}_model.h5')
        print("\n\n")

        return result

def main(args):
    trainer = LstmTrainer(args)
    return trainer.fit()
    