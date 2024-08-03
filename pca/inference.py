from datetime import datetime
import os

from .utils import json_processor
from .model import SSCModel as Model

import nltk
import numpy
from keras_preprocessing import sequence

class Inference:
    def __init__(self, args):
        self.model = Model(args)
        self.model.load_w2v_model()
        self.w2v_model = self.model.w2v_model
        self.word_vectors = self.model.word_vectors
        self.input_dir = args.input_dir
        self.lstm_model_save_dir = args.lstm_model_save_dir
        self.json_types = args.json_types
        self.max_len = 50

    def create_dataset(self,  datas, Xset, Yset):
        valid_words = set(self.word_vectors.key_to_index)
        debug_set = []
        for k in range(len(datas)):
            block = datas[k]    
            X, y = block
            token = (X)
            if not token:
                continue

            if len(debug_set) < 2:
                debug_set.extend([
                    [
                        word
                        for t in token 
                        if t and t != " " 
                        for word in nltk.word_tokenize(str(t)) 
                        if word in valid_words and word != " "
                    ]
                ])

            vectorlist = [
                self.w2v_model.wv[word].tolist() 
                for t in token 
                if t and t != " " 
                for word in nltk.word_tokenize(str(t)) 
                if word in valid_words and word != " "
            ]

            Xset.extend([vectorlist]) #append the list of vectors to the X (independent variable)
            Yset.extend([y]) #append the label to the Y (dependent variable)

        print("database：")
        print(debug_set)
        
        return Xset, Yset
        

    def predict(self):
        blocks = []
        dir_path = os.path.join(os.getcwd(), self.input_dir)
        
        blocks.extend(json_processor.traverse_and_read_json_with_label(dir_path, self.json_types))
      
        DataX = []
        DataY = []
        DataX, DataY = self.create_dataset(blocks, DataX, DataY)
        DataX = sequence.pad_sequences(DataX, padding='post', dtype='float32', maxlen=self.max_len)

        X_data =  numpy.array(DataX)
        y_data =  numpy.array(DataY)
        
        print("X_data.shape = ", X_data.shape)
        model = self.model.load_model(self.lstm_model_save_dir, X_data)

        y_prob, y_classes = self.model.predict(model, X_data)

        print("prob:", y_prob)
        print("class:", y_classes)
        print("ps: 1, bad; 0, good")
        count_zeros = len(y_classes) - numpy.count_nonzero(y_classes)
        print("Number of good:", count_zeros)
        count_ones = numpy.count_nonzero(y_classes)
        print("Number of bad:", count_ones)

        return y_classes


def main(args):
    begin = datetime.now()
    inference = Inference(args)
    result = inference.predict()
    end = datetime.now()
    duration = end - begin
    print(f"Duration of predict: {duration}\n\n")   
    return result
    