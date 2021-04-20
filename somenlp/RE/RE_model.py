import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

class REmodel():
    """Relation extraction model
    """
    def __init__(self, model_config, data_handler, output_handler):
        """Init

        Args:
            model_config (dict): configuration of the RE model
            data_handler (somenlp.NER.DataHandler)
            output_handler (somenlp.NER.OutputHandler)
        """
        self.data_handler = data_handler
        self.output_handler = output_handler
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

    def _train_model(self, data):
        """Train model based on provided data

        Args:
            data (list): features 
        """
        input_data = pd.DataFrame(data)
        y_train = input_data['label'].values
        X_train = input_data.loc[:, input_data.columns != 'label']
        self.X_train = X_train
        self.model.fit(X_train, y_train)

    def train(self):
        """Train RE model
        """
        all_train_samples = []
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['train']):
            print("Getting samples from {} dataset from train set".format(idx))
            all_train_samples.extend(dataset['relext_feature_list'])
        self._train_model(all_train_samples)

    def test(self):
        """Test RE model
        """
        out_s = ''
        cls_for_latex = {}
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['test']):
            print("Start testing on corpus {}".format(idx))
            input_data = pd.DataFrame(dataset['relext_feature_list'])
            y_test = input_data['label'].values
            X_test = input_data.loc[:, input_data.columns != 'label']
            predictions = self.model.predict(X_test)
            cm = confusion_matrix(y_test, predictions, labels=self.model.classes_)
            cl_no_negatives = classification_report(y_test, predictions, labels=self.model.classes_[self.model.classes_!='none'])
            cl_no_negatives_dict = classification_report(y_test, predictions, labels=self.model.classes_[self.model.classes_!='none'], output_dict=True)
            cls_for_latex[dataset['name']] = cl_no_negatives_dict
            cl_with_negatives = classification_report(y_test, predictions, labels=self.model.classes_)
            out_s += 'Test Corpus {}: {}\n\nConfusion Matrix:\n{}\n{}\n\nClassification without negative samples:\n{}\n\nClassification with negative samples:\n{}\n\n'.format(
                idx, 
                dataset['name'], 
                self.model.classes_, 
                cm,
                cl_no_negatives,
                cl_with_negatives
            )
        self.output_handler.cl_for_latex(cls_for_latex)
        print(out_s)
        with open('{}/rel_extraction_result.log'.format(self.output_handler.log_dir), 'w') as out_file:
            out_file.write(out_s)
    
    def show_features_importance(self):
        """Print a summary of the feature importance provided by sklearn
        """
        feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        print("Feature importance for classifier:")
        print(feat_importances.sort_values())
