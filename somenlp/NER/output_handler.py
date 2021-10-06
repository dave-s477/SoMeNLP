import json

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from sklearn.metrics import confusion_matrix
from articlenizer.formatting import bio_to_brat

class OutputHandler():
    def __init__(self, name, time='0_0_0', checkpoint={}, log_dir='logs', save_dir='save'):
        if 'model' in checkpoint and checkpoint['model']:
            self.model_loc = checkpoint['model']
        else:
            self.model_loc = ''

        if self.model_loc and 'log_dir' in checkpoint:
            self.log_dir = Path(checkpoint['log_dir'])
        else:
            self.log_dir = Path('{}/{}/{}'.format(log_dir, name, time))
        self.writer = SummaryWriter(self.log_dir)
        
        if self.model_loc and 'save_dir' in checkpoint:
            self.save_dir = Path(checkpoint['save_dir'])
        else:
            self.save_dir = Path('{}/{}/{}'.format(save_dir, name, time))
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_json(self, data, name='encoding'):
        with open('{}/{}.json'.format(self.save_dir, name), 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def load_encoding(self):
        with open('{}/encoding.json'.format(self.save_dir), 'r') as json_file:
            encoding_dict = json.load(json_file)
        return encoding_dict

    def print_scalars(self, scalars, epoch, name, meta_name=''):
        out_s = 'Classification result on {}{} ep {}:\n'.format(meta_name, name, epoch)
        for idx, (k, v) in enumerate(scalars.items()):
            if idx % 3 == 0:
                out_s += '\n'
            out_s += '{}:\t{}\n'.format(k, round(v, 3))
        out_s += 'Done\n\n'
        print(out_s)

    def write_scalars(self, scalars, epoch):
        for scalar_key, scalar_value in scalars.items():
            self.writer.add_scalar(scalar_key, scalar_value, epoch)

    def print_errors(self, labels, predictions, sentences, max_output_length, data_set_name, word2name):
        out_s = ''
        for sent_true, sent_pred, sent_coded_words in zip(labels, predictions, sentences):
            if sent_true != sent_pred: 
                out_s += "Wrong sentence:\n"
                output_string_true, output_string_pred, output_string_words = '', '', ''
                if isinstance(word2name, dict):
                    sentence_words = [word2name[w] for w in sent_coded_words]
                else:
                    sentence_words = word2name.convert_ids_to_tokens(sent_coded_words)
                for true_label, predicted_label, word in zip(sent_true, sent_pred, sentence_words):
                    next_length = max(len(word), len(true_label), len(predicted_label)) + 1
                    if (len(output_string_words) + next_length) > max_output_length:
                        out_s += 'Sent:\t{}\n'.format(output_string_words)
                        out_s += 'True:\t{}\n'.format(output_string_true)
                        out_s += 'Pred:\t{}\n'.format(output_string_pred)
                        output_string_true, output_string_pred, output_string_words = '', '', ''
                    output_string_words += '{:{}s}'.format(word, next_length)
                    output_string_true += '{:{}s}'.format(true_label, next_length)
                    output_string_pred += '{:{}s}'.format(predicted_label, next_length)

                if output_string_words:
                    out_s += 'Sent:\t{}\n'.format(output_string_words)
                    out_s += 'True:\t{}\n'.format(output_string_true)
                    out_s += 'Pred:\t{}\n'.format(output_string_pred)
                out_s += '\n\n'
        out_loc = self.save_dir / '{}_tagging_errors.txt'.format(data_set_name)
        with out_loc.open(mode='w') as out_f:
            out_f.write(out_s)
    
    def c_matrix(self, names, labels, predictions, tag_mode):
        tags = []
        for n in names:
            if n != 'O':
                tags.append('B-{}'.format(n))
                tags.append('I-{}'.format(n))
                if tag_mode == 'bioes':
                    tags.append('S-{}'.format(n))
                    tags.append('E-{}'.format(n))
        tags.append('O')
        t = [item for sublist in labels for item in sublist]
        p = [item for sublist in predictions for item in sublist]
        unique_tags = set(tags)
        cm = confusion_matrix(t, p, labels=tags)
        out_s = """
Confusion Matrix for:
{}
{}
        """.format(tags, cm)
        print(out_s)

    def save_predictions_fct(self, path, predictions, text):
        with path['out'].open(mode='w') as out_f:
            for preds in predictions:
                out_f.write('{}\n'.format(' '.join(preds).rstrip()))
        with path['out-text'].open(mode='w') as out_t:
            for line in text:
                out_t.write('{}\n'.format(' '.join(line).rstrip()))
    
    def save_predictions(self, path, predictions, text):
        if not isinstance(predictions, dict):
            self.save_predictions_fct(path, predictions, text)
        else:
            path_out = str(path['out'])
            for k, v in predictions.items():
                path['out'] = Path(path_out + '.' + k)
                self.save_predictions_fct(path, v, text)
        
    def summarize_predictions_fct(self, path, predictions, text):
        print("Predicted entities for {}".format(path.name))
        out_path = Path(str(path) + '.sum')
        entities, _, _ = bio_to_brat(text, predictions, split_sent=False, split_words=False)
        out_s = ''
        for e in entities:
            out_s += '{}\t{} {} {}\t{}\n'.format(e['id'], e['type'], e['beg'], e['end'], e['string'])
        with out_path.open(mode='w') as out_f:
            out_f.write(out_s)
        print(out_s)

    def summarize_predictions(self, path, predictions, text):
        if not isinstance(predictions, dict):
            self.summarize_predictions_fct(path['out'], predictions, text)
        else:
            for k, v in predictions.items():
                self.summarize_predictions_fct(Path(str(path['out']) + k), v, text)

    def cl_for_latex(self, dictionary, round_n=2):
        new_mapping = {}
        for k,v in dictionary.items():
            for label, values in v.items():
                if label not in new_mapping:
                    new_mapping[label] = {}
                for metric, result in values.items():
                    new_mapping[label][metric + '_' + k] = round(result, round_n)
        separator='&'
        s = ''
        for k,v in new_mapping.items():
            s += '{} {} {} ({}) {} {} ({}) {} {} ({}) {} {} ({})\n'.format(
                k, 
                separator,
                v['precision_test_0'],
                v['precision_devel_0'],
                separator,
                v['recall_test_0'],
                v['recall_devel_0'], 
                separator,
                v['f1-score_test_0'],
                v['f1-score_devel_0'],
                separator,
                v['support_test_0'],
                v['support_devel_0']
            )
        return s
