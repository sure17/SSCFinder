from .utils import json_processor
import itertools
import os
import nltk
from gensim.models import Word2Vec, KeyedVectors

def remove_quotes(data):
    if isinstance(data, str):
        return data.replace('"', ' ')
    elif isinstance(data, dict):
        return {remove_quotes(key): remove_quotes(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_quotes(item) for item in data]
    else:
        return data

class W2VTrainer:
    def __init__(self, args):
        self.w2v_data_input_dir = args.w2v_data_input_dir
        self.json_types = args.json_types
        self.json_save_path = args.json_save_path
        self.remove_quotes = args.remove_quotes
        self.json_edit_save_path = args.json_edit_save_path
        self.mincount = args.mincount
        self.iteration = args.iteration
        self.embedding_dim = args.embedding_dim
        self.w2v_model_save_dir = args.w2v_model_save_dir
        self.overwrite_model = args.overwrite_model
        self.train_single = args.train_single
        self.single_mincount = args.single_mincount
        self.single_iteration = args.single_iteration
        self.single_embedding_dim = args.single_embedding_dim
        self.w2v_model_name = args.w2v_model_name
        self.kv_type = args.kv_type
        self.w2v_worker_num = args.w2v_worker_num

    def fetch_json(self):
        json_data = []
        
        for folder_name in self.w2v_data_input_dir:
            dir_path = os.path.join(os.getcwd(), folder_name)
            json_data.extend(json_processor.traverse_and_read_json(dir_path, self.json_types, self.kv_type))
        
        json_processor.save_json_to_file(json_data, self.json_save_path)

    def process_data(self):
        processing_funcs = []
        if self.remove_quotes:
            processing_funcs.append(remove_quotes)

        json_processor.process_json(self.json_save_path, self.json_edit_save_path, processing_funcs)

    def train_w2v_model(self):
        print("Loading...")
        with open(self.json_edit_save_path) as f:
            pythondata = f.read().lower().replace('\n', ' ')
            
        print(f"Length of the training file: {len(pythondata)}.")
        print(f"It contains {pythondata.count(' ')} individual code tokens.")

        print("now processing...")
        processed = pythondata
        all_sentences = nltk.sent_tokenize(processed)
        all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
        print("processed.\n")

        os.makedirs(self.w2v_model_save_dir, exist_ok=True)

        if self.train_single:
            min_count = self.single_mincount
            iterations = self.single_iteration
            vector_size = self.single_embedding_dim


            print(f"\n\nW2V model with min count {min_count} and {iterations} iterations and size {vector_size}")
            fname = os.path.join(self.w2v_model_save_dir, f"word2vec_{self.w2v_model_name}-{min_count}-{iterations}-{vector_size}.model")

            if os.path.isfile(fname):
                os.remove(fname)
            
            print("calculating model...")
            model = Word2Vec(all_words, vector_size=vector_size, min_count=min_count, epochs=iterations, workers=self.w2v_worker_num)
            model.save(fname)
        else:
            for min_count, iterations, vector_size in itertools.product(self.mincount, self.iteration, self.embedding_dim):
                print(f"\n\nW2V model with min count {min_count} and {iterations} iterations and size {vector_size}")
                fname = os.path.join(self.w2v_model_save_dir, f"word2vec_{self.w2v_model_name}-batch-{min_count}-{iterations}-{vector_size}.model")

                if os.path.isfile(fname):
                    if self.overwrite_model:
                        os.remove(fname)
                    else:
                        continue    
                
                print("calculating model...")
                model = Word2Vec(all_words, vector_size=vector_size, min_count=min_count, epochs=iterations, workers=self.w2v_worker_num)
                model.save(fname)

    def fit(self):
        if not self.overwrite_model and self.train_single:
            min_count = self.single_mincount
            iterations = self.single_iteration
            vector_size = self.single_embedding_dim

            fname = os.path.join(self.w2v_model_save_dir, f"word2vec_{self.w2v_model_name}-{min_count}-{iterations}-{vector_size}.model")

            if os.path.isfile(fname):
                print("pass")
                return
            
        self.fetch_json()
        self.process_data()
        self.train_w2v_model()


def main(args):
    trainer = W2VTrainer(args)
    trainer.fit()