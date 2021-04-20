import torch
import time

from .seqeval_custom import precision_recall_fscore_support

class Trainer():
    def __init__(self, device, model_wrapper, data_handler, output_handler, train_conf):
        self.device = device
        self.model_w = model_wrapper
        self.data_handler = data_handler
        self.output_handler = output_handler
        self.train_config = train_conf

    def _weighted_averages(self, support, *arrays):
        res = []
        for arr in arrays:
            weighted_average = sum([a * b for a, b in zip(support, arr)]) / sum(support)
            res.append(weighted_average)
        return res

    def _eval(self, labels, predictions, data_set_name, loss):
        precision_all, recall_all, fscore_all, support, names = precision_recall_fscore_support(labels, predictions, average=None)
        w_precision, w_recall, w_fscore = self._weighted_averages(support, precision_all, recall_all, fscore_all)
        scalars = {}
        for p, r, f, n in zip(precision_all, recall_all, fscore_all, names):
            scalars['{}/Precision/{}'.format(n, data_set_name)] = p
            scalars['{}/Recall/{}'.format(n, data_set_name)] = r
            scalars['{}/FScore/{}'.format(n, data_set_name)] = f
        scalars['Total/Precision/{}'.format(data_set_name)] = w_precision
        scalars['Total/Recall/{}'.format(data_set_name)] = w_recall
        scalars['Total/FScore/{}'.format(data_set_name)] = w_fscore
        scalars['Total/Loss/{}'.format(data_set_name)] = loss

        self.output_handler.print_scalars(scalars, self.model_w.global_epoch, data_set_name)
        self.output_handler.write_scalars(scalars, self.model_w.global_epoch)

        self.output_handler.c_matrix(names, labels, predictions, self.train_config['tag_mode'])

    def _train_model(self, train_loader, epochs):
        print("Starting training")
        for ep in range(1, epochs+1):
            self.model_w.model.train()
            print("Epoch {}".format(self.model_w.global_epoch))
            start = time.time()
            ep_loss, running_batch_loss, running_batch_count = 0, 0, 0
            for step, batch in enumerate(train_loader):
                running_batch_count += 1
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                self.model_w.optim.zero_grad()
                if self.model_w.optim_grouped_params is None:
                    loss = self.model_w.model.neg_log_likelihood(batch['tags'], char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                else:
                    outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                    loss = outputs[0]
                loss.backward()
                if self.model_w.optim_grouped_params is None:
                    self.model_w.optim.step()
                else:
                    self.model_w.optim.step()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model_w.model.parameters(), max_norm=self.model_w.config['model']['gen']['max_grad_norm'])
                    self.model_w.scheduler.step()
                ep_loss += loss.item()
                running_batch_loss += loss.item()

                if step > 0:
                    if step % self.train_config['print_batches'] == 0:
                        print("At batch {}".format(step))
                        print("Average loss over last batches: {}".format(running_batch_loss / running_batch_count))
                        running_batch_count = 0
                        running_batch_loss = 0
                    if step % self.train_config['save_batches'] == 0:
                        self.model_w.save_checkpoint(step)
                    if step % self.train_config['test_batches'] == 0:
                        self._test_model()
                    
            end = time.time()
            print("Epoch took {} seconds".format(round(end - start, 3)))

            if self.model_w.global_epoch % self.train_config['test_epochs'] == 0:
                self._test_model()
            if self.model_w.global_epoch % self.train_config['save_epochs'] == 0:
                self.model_w.save_checkpoint()

            self.model_w.global_epoch += 1

    def _test_model(self):
        self.model_w.model.eval()
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['test']):
            print("Start testing on corpus {}".format(idx))
            ep_loss = 0
            predictions, true_labels, input_masks, input_ids = [], [], [], []
            start = time.time()
            for step, batch in enumerate(dataset['dataloader']):
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                with torch.no_grad():
                    if self.model_w.optim_grouped_params is None:
                        tag_seq, score, input_mask = self.model_w.model(char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                    else:
                        outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                        logits = outputs[1]
                        tag_seq = torch.argmax(logits, axis=2)
                        input_mask = (
                            (batch['ids'] != self.data_handler.special_toks['cls_tok']) &
                            (batch['ids'] != self.data_handler.special_toks['pad_tok']) &
                            (batch['ids'] != self.data_handler.special_toks['sep_tok'])
                        )
                predictions.extend(tag_seq.tolist())
                true_labels.extend(batch['tags'].tolist())
                input_masks.extend(input_mask.tolist())
                input_ids.extend(batch['ids'].tolist())
            
            end = time.time()
            print("Testing on corpus {} took {} seconds".format(idx, round(end - start, 3)))

            sentences = []
            pred_tags = []
            valid_tags = []
            for j_p, j_t, j_s, j_m in zip(predictions, true_labels, input_ids, input_masks):
                pred_tags.append([])
                valid_tags.append([])
                sentences.append([])
                for i_p, i_t, i_s, i_m in zip(j_p, j_t, j_s, j_m):
                    if i_m:
                        pred_tags[-1].append(self.data_handler.encoding['tag2name'][i_p])
                        valid_tags[-1].append(self.data_handler.encoding['tag2name'][i_t])
                        sentences[-1].append(i_s)  
            
            self._eval(valid_tags, pred_tags, dataset['name'], ep_loss)

            if self.train_config['print_errors']:
                print(len(self.data_handler.encoding['word2name']))
                token_convert = self.data_handler.encoding['word2name'] if self.data_handler.tokenizer is None else self.data_handler.tokenizer
                self.output_handler.print_errors(valid_tags, pred_tags, sentences, self.train_config['max_output_length'], dataset['name'], token_convert)

    def train(self):
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['train']):
            print("Training on {} dataset from train set".format(idx))
            if dataset["epochs"] > 0:
                self.model_w.set_optim(dataset['optimizer'])
                if self.model_w.optim_grouped_params is not None:
                    self.model_w.set_scheduler((len(dataset['dataloader']) * dataset['epochs']), dataset['scheduler']['warm_up'])
                self._train_model(dataset['dataloader'], dataset['epochs'])

    def prediction(self, bio=True, summary=True):
        self.model_w.model.eval()
        start = time.time()
        iterator = self.data_handler.stream_files()
        for out_path, data_loader, text in iterator:
            ids, predictions, input_masks = [], [], []
            for batch in data_loader:
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                with torch.no_grad():
                    if self.model_w.optim_grouped_params is None:
                        tag_seq, score, input_mask = self.model_w.model(char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                    else:
                        outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                        logits = outputs[1]
                        tag_seq = torch.argmax(logits, axis=2)
                        input_mask = (
                            (batch['ids'] != self.data_handler.special_toks['cls_tok']) &
                            (batch['ids'] != self.data_handler.special_toks['pad_tok']) &
                            (batch['ids'] != self.data_handler.special_toks['sep_tok'])
                        )
                ids.extend(batch['ids'].tolist())
                predictions.extend(tag_seq.tolist())
                input_masks.extend(input_mask.tolist())

            pred_tags = []
            n_text = []
            for j, j_p, j_m in zip(ids, predictions, input_masks):
                pred_tags.append([])
                n_text.append([])
                for i, i_p, i_m in zip(j, j_p, j_m):
                    if i_m:
                        pred_tags[-1].append(self.data_handler.encoding['tag2name'][i_p])
                        n_text[-1].append(i)

            if self.data_handler.tokenizer is None:
                n_text = [[self.data_handler.encoding['word2name'][word] for word in sent] for sent in n_text]
            else:
                n_text = [[self.data_handler.tokenizer.convert_ids_to_tokens(word) for word in sent] for sent in n_text]

            if bio:
                self.output_handler.save_predictions(out_path, pred_tags, n_text)
            if summary:
                self.output_handler.summarize_predictions(out_path, pred_tags, n_text)

        end = time.time()
        print("Predicting all files took {} seconds".format(round(end - start, 3)))
