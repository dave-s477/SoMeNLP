import pandas as pd
import numpy as np
import pickle
from torch import cross 

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from itertools import combinations

class REmodel():
    """Relation extraction model
    """
    def __init__(self, gen_config, model_config, data_handler, output_handler, output_config, feature_generator=None):
        """Init

        Args:
            model_config (dict): configuration of the RE model
            data_handler (somenlp.NER.DataHandler)
            output_handler (somenlp.NER.OutputHandler)
        """
        self.gen_config = gen_config
        self.model_config = model_config
        self.data_handler = data_handler
        self.output_handler = output_handler
        self.feature_generator = feature_generator
        self.output_config = output_config
        if gen_config['type'] == 'RF':
            self.model = RandomForestClassifier(
                n_estimators=model_config["n_estimators"],
                criterion=model_config["criterion"],
                max_depth=model_config["max_depth"],
                min_samples_split=model_config["min_samples_split"],
                min_samples_leaf=model_config["min_samples_leaf"],
                min_weight_fraction_leaf=model_config["min_weight_fraction_leaf"],
                max_features=model_config["max_features"],
                max_leaf_nodes=model_config["max_leaf_nodes"],
                max_samples=model_config["max_samples"]
            )
        elif gen_config['type'] == 'NN':
            if model_config['scaler']:
                self.scaler = StandardScaler()
            self.model = MLPClassifier(
                solver = model_config['solver'],
                batch_size = gen_config['batch_size'],
                max_iter = model_config['epochs'],
                learning_rate_init = model_config['lr'],
                hidden_layer_sizes=tuple(model_config['layers'])
            )
        else:
            raise(RuntimeError("Got unsupported model type {}".format(model_config['type'])))

    def _train_model(self, data):
        """Train model based on provided data

        Args:
            data (list): features 
        """
        input_data = pd.DataFrame(data)
        y_train = input_data['label'].values
        X_train = input_data.loc[:, input_data.columns != 'label']
        if self.gen_config['type'] == 'NN' and self.model_config['scaler']:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
        #self.X_train = X_train
        self.model.fit(X_train, y_train)

    def train(self):
        """Train RE model
        """
        all_train_samples = []
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['train']):
            print("Getting samples from {} dataset from train set".format(idx))
            all_train_samples.extend(dataset['relext_feature_list'])
        self._train_model(all_train_samples)   

    def cross_val(self):
        all_train_samples = []
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['train']):
            print("Getting samples from {} dataset from train set".format(idx))
            all_train_samples.extend(dataset['relext_feature_list'])
        input_data = pd.DataFrame(all_train_samples)
        y = input_data['label'].values
        X = input_data.loc[:, input_data.columns != 'label'] 
        if self.gen_config['type'] == 'NN' and self.model_config['scaler']:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        cv_results = cross_validate(self.model, X, y, cv=5, scoring=('precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro'))
        for k, v in cv_results.items(): 
            if k.startswith('test'):
                print('{}: {}'.format(k, v))

    def test(self):
        """Test RE model
        """
        out_s = ''
        cls_for_latex = {}
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['test']):
            print("Start testing on corpus {}".format(idx))
            out_s += 'Test Corpus {}: {}\n\n'.format(idx, dataset['name'])
            input_data = pd.DataFrame(dataset['relext_feature_list'])
            y_test = input_data['label'].values
            X_test = input_data.loc[:, input_data.columns != 'label']
            if self.gen_config['type'] == 'NN' and self.model_config['scaler']:
                X_test = self.scaler.transform(X_test)
            predictions = self.model.predict(X_test)
            if "confusion" in self.output_config and self.output_config['confusion']:
                cm = confusion_matrix(y_test, predictions, labels=self.model.classes_)
                out_s += "Confusion Matrix:\n{}\n{}\n\n".format(self.model.classes_, cm)
            if "no_neg" in self.output_config and self.output_config['no_neg']:
                cl_no_negatives = classification_report(y_test, predictions, labels=self.model.classes_[self.model.classes_!='none'])
                out_s += "Classification without negative samples:\n{}\n\n".format(cl_no_negatives)
            if "with_neg" in self.output_config and self.output_config['with_neg']:
                cl_with_negatives = classification_report(y_test, predictions, labels=self.model.classes_)
                out_s += "Classification with negative samples:\n{}\n\n".format(cl_with_negatives)
            if "latex" in self.output_config and self.output_config['latex']:
                cl_no_negatives_dict = classification_report(y_test, predictions, labels=self.model.classes_[self.model.classes_!='none'], output_dict=True)
                cls_for_latex[dataset['name']] = cl_no_negatives_dict
        if "latex" in self.output_config and self.output_config['latex']:
            out_s += "Formatted for latex: {}".format(self.output_handler.cl_for_latex(cls_for_latex))
        
        print(out_s)
        if "save_log" in self.output_config and self.output_config['save_log']:
            with open('{}/rel_extraction_result.log'.format(self.output_handler.log_dir), 'w') as out_file:
                out_file.write(out_s)

    def predict(self, output):
        print("Starting prediction")
        iterator = self.feature_generator.stream_files()
        for input in iterator:
            prediction_outputs = []
            for idx, (sentence, features, entities) in enumerate(zip(input['sentences'], input['relext_feature_list'], input['entity_list'])):
                prediction_outputs.append([])
                if features is not None:
                    # print(sentence)
                    # print(input['tags'])
                    # print()
                    #print(features)
                    X = pd.DataFrame(features)
                    if self.gen_config['type'] == 'NN' and self.model_config['scaler']:
                        X = self.scaler.transform(X)
                    predictions = self.model.predict(X)
                    #print(X)
                    for ent_pair, assigned_label in zip(entities, predictions):
                        if assigned_label != 'none':
                            prediction_outputs[-1].append([ent_pair, assigned_label])
                    #print(predictions)
                    #print()
            with input['out_name'].open(mode='w') as out_f:
                for line_pred in prediction_outputs:
                    line_string = ''
                    for pred in line_pred:
                        line_string += '{}\t{}\t{}\t{}\t{}\t{}\t{};;'.format(pred[1], pred[0][0]['string'], pred[0][0]['beg'], pred[0][0]['idx'], pred[0][1]['string'], pred[0][1]['beg'], pred[0][1]['idx'])
                    out_f.write(line_string + '\n')
            #prediction_outputs
    
    def show_features_importance(self):
        """Print a summary of the feature importance provided by sklearn
        """
        feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        print("Feature importance for classifier:")
        print(feat_importances.sort_values())

    def save(self):
        output_name = '{}/model.sav'.format(self.output_handler.save_dir)
        pickle.dump(self.model, open(output_name, 'wb'))
        print("Saved model at {}".format(output_name))

    def load(self, checkpoint):
        self.model = pickle.load(open(checkpoint['model'], 'rb'))
        print("Loaded model from {}".format(checkpoint['model']))
