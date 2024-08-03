import os

from gensim.models import Word2Vec
import keras
from keras import backend as K
from memory_profiler import profile
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Activation, Dot, Concatenate, Conv1D, MaxPooling1D, Flatten

@profile
def predict_memory_usage(model, data):
    return model.predict(data)

class OCN(Model):
    def __init__(self, time_steps, input_dim):
        super(OCN, self).__init__()

        self.time_steps = time_steps
        self.input_dim = input_dim

        backbone = VGG16(weights='imagenet', include_top=False, input_shape=(time_steps, input_dim, 3))
        for layer in backbone.layers[:-2]:
            layer.trainable = False
        self.backbone = tf.keras.Sequential(backbone.layers[:-2])

        self.classifier = tf.keras.Sequential([
            Dense(1024, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])


    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)
        inputs = tf.tile(inputs, multiples=[1, 1, 1, 3]) 
        inputs = tf.image.resize(inputs, (self.time_steps, self.input_dim))

        x = self.backbone(inputs)
        x = tf.nn.relu(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.classifier(x)

        return x

class SSCModel:
    def __init__(self, args):
        self.w2v_model_name = args.w2v_model_name if hasattr(args, 'w2v_model_name') else None
        self.w2v_model_save_dir = args.w2v_model_save_dir if hasattr(args, 'w2v_model_save_dir') else None
        self.w2v_mincount = args.w2v_mincount if hasattr(args, 'w2v_mincount') else None
        self.w2v_epochs = args.w2v_epochs if hasattr(args, 'w2v_epochs') else None
        self.w2v_embedding_dim = args.w2v_embedding_dim if hasattr(args, 'w2v_embedding_dim') else None
        self.input_dir_good = args.input_dir_good if hasattr(args, 'input_dir_good') else None
        self.input_dir_bad = args.input_dir_bad if hasattr(args, 'input_dir_bad') else None
        self.json_types = args.json_types if hasattr(args, 'json_types') else None
        self.train_ratio = args.train_ratio if hasattr(args, 'train_ratio') else None
        self.test_ratio = args.test_ratio if hasattr(args, 'test_ratio') else None
        self.val_ratio = args.val_ratio if hasattr(args, 'val_ratio') else None
        self.dropout = args.dropout if hasattr(args, 'dropout') else None
        self.neurons = args.neurons if hasattr(args, 'neurons') else None 
        self.optimizer = args.optimizer if hasattr(args, 'optimizer') else None
        self.epochs = args.epochs if hasattr(args, 'epochs') else None
        self.batchsize = args.batchsize if hasattr(args, 'batchsize') else None
        self.enable_class_weights = args.enable_class_weights if hasattr(args, 'enable_class_weights') else None
        self.lstm_model_save_dir = args.lstm_model_save_dir if hasattr(args, 'lstm_model_save_dir') else None
        self.model_type = args.model if hasattr(args, 'model') else None
        self.w2v_model_name = args.w2v_model_name if hasattr(args, 'w2v_model_name') else None


        self.max_len = 50
        self.models = {
            "lstm" : self.build_lstm(),
            "lstm-attention" : self.build_lstm_attention(),
            "cnn" : self.build_cnn(),
            "dnn" : self.build_dnn(),
            "rcnn" : self.build_rcnn(),
            "dpcnn" : self.build_dpcnn(),
            "ocn" : self.build_ocn()
        }


    def get_evaluate_func(self):
        def f1_loss(y_true, y_pred):
            y_true = K.cast(y_true, 'float')
            y_pred = K.cast(y_pred, 'float')

            tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
            tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
            fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
            fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

            p = tp / (tp + fp + K.epsilon())
            r = tp / (tp + fn + K.epsilon())

            f1 = 2*p*r / (p+r+K.epsilon())
            f1 = tf.where(tf.math.is_nan(f1), K.zeros_like(f1), f1)
            return 1 - K.mean(f1)

        def f1(y_true, y_pred):
            y_true = K.cast(y_true, 'float')
            y_pred = K.cast(y_pred, 'float')

            def recall(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall
            def precision(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        return f1_loss, [f1]
    
    def load_model(self, model_weight_path, input):
        f1_loss, f1 = self.get_evaluate_func()

        model = self.models[self.model_type]
        model.compile(loss=f1_loss, optimizer=self.optimizer, metrics=f1)
        model_weight_path = os.path.join(model_weight_path, f'{self.model_type}_model.h5')
        print("model_weight: ", model_weight_path)
        model(input)
        model.load_weights(model_weight_path)
        return model

    def load_w2v_model(self):
        # get word2vec model
        w2v_file_name = os.path.join(self.w2v_model_save_dir, f"word2vec_{self.w2v_model_name}-{self.w2v_mincount}-{self.w2v_epochs}-{self.w2v_embedding_dim}.model")
        print("w2v: ", w2v_file_name)
        
        #load word2vec model
        if not (os.path.isfile(w2v_file_name)):
            print("word2vec model is still being created...")
            return None

        w2v_model = Word2Vec.load(w2v_file_name)
        self.w2v_model = w2v_model
        self.word_vectors = self.w2v_model.wv

        print("w2v, mincount = %d, epoch = %d, dim = %d ..." % (self.w2v_mincount, self.w2v_epochs, self.w2v_embedding_dim))
        return w2v_model
    
    def predict(self, model, X_data):
        yhat_prob = predict_memory_usage(model, X_data)
        yhat_prob = model.predict(X_data, verbose=0)
        yhat_classes = np.int64(yhat_prob > 0.5)
        return yhat_prob, yhat_classes

    def evaluate(self, model, X_data, y_data):
        yhat_prob, yhat_classes = self.predict(model, X_data)
        accuracy = accuracy_score(y_data, yhat_classes)
        precision = precision_score(y_data, yhat_classes)
        recall = recall_score(y_data, yhat_classes)
        F1Score = f1_score(y_data, yhat_classes)
        fpr, tpr, thresholds = roc_curve(y_data, yhat_prob)
        return accuracy, precision, recall, F1Score, fpr, tpr
    

    def build_lstm(self):
        model = Sequential()
        model.add(LSTM(self.neurons, dropout = self.dropout, recurrent_dropout = self.dropout)) #around 50 seems good
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_lstm_attention(self):
        def attention(inputs):
            hidden_state = inputs[0]  # Shape: (batch_size, time_steps, hidden_dim)
            context = inputs[1]  # Shape: (batch_size, hidden_dim)

            # Calculate attention weights
            attention_weights = Dot(axes=[2, 1])([hidden_state, context])
            attention_weights = Activation('softmax')(attention_weights)

            # Apply attention weights to hidden states
            weighted_hidden_state = Dot(axes=[1, 1])([attention_weights, hidden_state])
            return weighted_hidden_state

        # Define the model
        time_steps = self.max_len
        input_dim = self.w2v_embedding_dim
        inputs = Input(shape=(time_steps, input_dim))
        lstm_output = LSTM(self.neurons, dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True)(inputs)
        attention_output = attention([lstm_output, lstm_output[:, -1, :]])
        output = Dense(1, activation='sigmoid')(attention_output)

        model = Model(inputs=inputs, outputs=output)
        return model

    def build_cnn(self):
        filters = 32
        kernel_size = 3
        pool_size = 2

        model = Sequential()
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size))
        model.add(Flatten())
        model.add(Dense(self.neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_dnn(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(self.neurons, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_rcnn(self):
        time_steps = self.max_len
        input_dim = self.w2v_embedding_dim
        filters = 32
        kernel_size = 3

        # Define the model
        model = Sequential()
        model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(time_steps, input_dim)))
        model.add(LSTM(self.neurons, dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True))
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def build_dpcnn(self):
        filters = 250
        kernel_size = 3
        pool_size = 3
        input_dim = self.w2v_embedding_dim

        inputs = Input(shape=(self.max_len, input_dim))
        conv1 = Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)
        conv2 = Conv1D(filters, kernel_size, activation='relu', padding='same')(conv1)
        maxpool = MaxPooling1D(pool_size=pool_size, strides=1)(conv2)

        # Residual connection
        conv_res = Conv1D(filters, 1, padding='same')(maxpool)
        shortcut = Conv1D(filters, 1, padding='same')(maxpool)
        added = Concatenate(axis=-1)([conv_res, shortcut])

        # Stack of blocks
        conv_block = added
        for _ in range(15):
            conv1 = Conv1D(filters, kernel_size, activation='relu', padding='same')(conv_block)
            conv2 = Conv1D(filters, kernel_size, activation='relu', padding='same')(conv1)
            maxpool = MaxPooling1D(pool_size=pool_size, strides=1)(conv2)
            conv_res = Conv1D(filters, 1, padding='same')(maxpool)
            added = Concatenate(axis=-1)([conv_res, maxpool])
            conv_block = added

        flatten = Flatten()(conv_block)
        output = Dense(1, activation='sigmoid')(flatten)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_ocn(self):
        time_steps = self.max_len
        input_dim = self.w2v_embedding_dim
        return OCN(time_steps=time_steps, input_dim=input_dim)
    
    
