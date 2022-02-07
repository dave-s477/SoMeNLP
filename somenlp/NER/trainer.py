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
            if sum(support) == 0:
                weighted_average = 0
            else:
                weighted_average = sum([a * b for a, b in zip(support, arr)]) / sum(support)
            res.append(weighted_average)
        return res

    def _eval_fct(self, labels, predictions, data_set_name, loss, meta_name=''):
        precision_all, recall_all, fscore_all, support, names = precision_recall_fscore_support(labels, predictions, average=None)
        w_precision, w_recall, w_fscore = self._weighted_averages(support, precision_all, recall_all, fscore_all)
        scalars = {}
        for p, r, f, n in zip(precision_all, recall_all, fscore_all, names):
            scalars['{}{}/Precision/{}'.format(meta_name, n, data_set_name)] = p
            scalars['{}{}/Recall/{}'.format(meta_name, n, data_set_name)] = r
            scalars['{}{}/FScore/{}'.format(meta_name, n, data_set_name)] = f

            if (not meta_name or meta_name.rstrip('/') == 'software') and data_set_name == self.train_config['eval_dataset_name'] and n == 'Application':
                self.model_w.current_performance = f
                if self.model_w.best_performance <= f:
                    self.model_w.current_is_best = True
                    self.model_w.best_performance = f

        scalars['{}Total/Precision/{}'.format(meta_name, data_set_name)] = w_precision
        scalars['{}Total/Recall/{}'.format(meta_name, data_set_name)] = w_recall
        scalars['{}Total/FScore/{}'.format(meta_name, data_set_name)] = w_fscore
        scalars['{}Total/Loss/{}'.format(meta_name, data_set_name)] = loss

        self.output_handler.print_scalars(scalars, self.model_w.global_epoch, data_set_name, meta_name)
        self.output_handler.write_scalars(scalars, self.model_w.global_epoch)

        self.output_handler.c_matrix(names, labels, predictions, self.train_config['tag_mode'])

    def _eval(self, labels, predictions, data_set_name, loss):
        if not isinstance(labels, dict):
            self._eval_fct(labels, predictions, data_set_name, loss)
        else:
            for k in labels.keys():
                self._eval_fct(labels[k], predictions[k], data_set_name, loss, meta_name=k+'/')

    def _get_train_depth(self, ep, hierarchy, max_depth=3):
        for _, v in hierarchy.items():
            if ep <= v['limit']:
                return v['depth']
        return max_depth

    def _train_model(self, train_loader, epochs):
        print("Starting training")
        for ep in range(1, epochs+1):
            self.model_w.model.train()
            self.model_w.current_is_best = False
            print("Epoch {}".format(self.model_w.global_epoch))
            if self.data_handler.multi_task_mapping:
                if 'hierarchy_depth' in self.model_w.config['model']['gen']:
                    train_depth = self._get_train_depth(ep, self.model_w.config['model']['gen']['hierarchy_depth'])
                else:
                    train_depth = 4
                print("Training multi-label model with max depth {}".format(train_depth))
            start = time.time()
            ep_loss, running_batch_loss, running_batch_count = 0, 0, 0
            for step, batch in enumerate(train_loader):
                running_batch_count += 1
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                if self.model_w.optim_grouped_params is None:
                    loss = self.model_w.model.neg_log_likelihood(batch['tags'], char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                else:
                    if not self.data_handler.multi_task_mapping:
                        outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                        loss = outputs[0]
                    else:
                        if len(self.data_handler.encoding['tag2idx']) == 4:
                            outputs = self.model_w.model(
                                batch['ids'], 
                                token_type_ids=None, 
                                attention_mask=batch['masks'], 
                                software_labels=batch['software'],
                                soft_type_labels=batch['soft_type'],
                                mention_type_labels = batch['mention_type'],
                                soft_purpose_labels=batch['soft_purpose'],
                                sequence_lengths=batch['lengths'],
                                train_depth=train_depth,
                                teacher_forcing=True)
                        elif len(self.data_handler.encoding['tag2idx']) == 3:
                            outputs = self.model_w.model(
                                batch['ids'], 
                                token_type_ids=None, 
                                attention_mask=batch['masks'], 
                                software_labels=batch['software'],
                                soft_type_labels=batch['soft_type'],
                                soft_purpose_labels=batch['soft_purpose'],
                                sequence_lengths=batch['lengths'],
                                train_depth=train_depth,
                                teacher_forcing=True)
                        else:
                            raise(RuntimeError("Unsupported data transformation configuration"))
                        loss = outputs[0]
                loss.backward()
                if self.model_w.optim_grouped_params is None:
                    self.model_w.optim.step()
                    self.model_w.optim.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model_w.model.parameters(), max_norm=self.model_w.config['model']['gen']['max_grad_norm'])
                    self.model_w.optim.step()
                    self.model_w.optim.zero_grad()
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
            if self.model_w.global_epoch >= self.train_config['save_from']:
                if self.model_w.global_epoch % self.train_config['save_epochs'] == 0:
                    self.model_w.save_checkpoint()
                if 'save_max' in self.train_config and self.train_config['save_max'] and self.model_w.current_is_best:
                    print("Saving model with best performance..")
                    self.model_w.save_checkpoint()

            self.model_w.global_epoch += 1

    def _test_model(self):
        self.model_w.model.eval()
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['test']):
            print("Start testing on corpus {}".format(idx))
            ep_loss = 0
            if not self.data_handler.multi_task_mapping:
                predictions, true_labels, input_masks, input_ids = [], [], [], []
            else:
                predictions, true_labels = {}, {}
                for k in self.data_handler.tag_remapping.keys():
                    predictions[k] = []
                    true_labels[k] = []
                input_masks, input_ids = [], []
            start = time.time()
            for step, batch in enumerate(dataset['dataloader']):
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                with torch.no_grad():
                    if self.model_w.optim_grouped_params is None:
                        tag_seq, score, input_mask = self.model_w.model(char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                        predictions.extend(tag_seq.tolist())
                        true_labels.extend(batch['tags'].tolist())
                    else:
                        if not self.data_handler.multi_task_mapping:
                            outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                            logits = outputs[1]
                            tag_seq = torch.argmax(logits, axis=2)
                            predictions.extend(tag_seq.tolist())
                            true_labels.extend(batch['tags'].tolist())
                        else:
                            if len(self.data_handler.encoding['tag2idx']) == 4:
                                outputs = self.model_w.model(
                                    batch['ids'], 
                                    token_type_ids=None, 
                                    attention_mask=batch['masks'], 
                                    software_labels=batch['software'],
                                    soft_type_labels=batch['soft_type'],
                                    mention_type_labels = batch['mention_type'],
                                    soft_purpose_labels=batch['soft_purpose'],
                                    sequence_lengths=batch['lengths'],
                                    train_depth=3,
                                    teacher_forcing=False)
                                logits = {
                                    'software': outputs[1],
                                    'soft_type': outputs[2],
                                    'mention_type': outputs[3],
                                    'soft_purpose': outputs[4]
                                }
                            elif len(self.data_handler.encoding['tag2idx']) == 3:
                                outputs = self.model_w.model(
                                    batch['ids'], 
                                    token_type_ids=None, 
                                    attention_mask=batch['masks'], 
                                    software_labels=batch['software'],
                                    soft_type_labels=batch['soft_type'],
                                    soft_purpose_labels=batch['soft_purpose'],
                                    sequence_lengths=batch['lengths'],
                                    train_depth=3,
                                    teacher_forcing=False)
                                logits = {
                                    'software': outputs[1],
                                    'soft_type': outputs[2],
                                    'soft_purpose': outputs[3]
                                }
                            else:
                                raise(RuntimeError("Unsupported data transformation configuration"))
                            for k in predictions.keys():
                                if self.model_w.model_type in ['MultiSciBERTCRF', 'MultiOpt2SciBERTCRF']:
                                    predictions[k].extend(logits[k].tolist())
                                else:
                                    predictions[k].extend(torch.argmax(logits[k], axis=2).tolist())
                                true_labels[k].extend(batch[k].tolist())

                input_mask = (
                    (batch['ids'] != self.data_handler.special_toks['cls_tok']) &
                    (batch['ids'] != self.data_handler.special_toks['pad_tok']) &
                    (batch['ids'] != self.data_handler.special_toks['sep_tok'])
                )
                input_masks.extend(input_mask.tolist())
                input_ids.extend(batch['ids'].tolist())
            
            end = time.time()
            print("Testing on corpus {} took {} seconds".format(idx, round(end - start, 3)))

            if not self.data_handler.multi_task_mapping:
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
            else:
                sentences = []
                pred_tags, valid_tags = {}, {}
                for k in self.data_handler.encoding['tag2idx'].keys():
                    pred_tags[k] = []
                    valid_tags[k] = []
                for top_idx, (j_s, j_m) in enumerate(zip(input_ids, input_masks)):
                    for k in pred_tags.keys():
                        pred_tags[k].append([])
                        valid_tags[k].append([])
                    sentences.append([])
                    for bottom_idx, (i_s, i_m) in enumerate(zip(j_s, j_m)):
                        if i_m:
                            sentences[-1].append(i_s)
                            for k in pred_tags.keys():
                                pred_tags[k][-1].append(self.data_handler.encoding['tag2name'][k][predictions[k][top_idx][bottom_idx]])
                                valid_tags[k][-1].append(self.data_handler.encoding['tag2name'][k][true_labels[k][top_idx][bottom_idx]])
            
            self._eval(valid_tags, pred_tags, dataset['name'], ep_loss)

            if self.train_config['print_errors']:
                token_convert = self.data_handler.encoding['word2name'] if self.data_handler.tokenizer is None else self.data_handler.tokenizer
                self.output_handler.print_errors(valid_tags, pred_tags, sentences, self.train_config['max_output_length'], dataset['name'], token_convert)

    def train(self):
        for idx, dataset in enumerate(self.data_handler.data_config['sets']['train']):
            print("Training on {} dataset from train set".format(idx))
            if dataset["epochs"] > 0:
                self.model_w.set_optim(dataset['optimizer'])
                if self.model_w.optim_grouped_params is not None:
                    self.model_w.set_scheduler((len(dataset['dataloader']) * dataset['epochs']), dataset['scheduler'])
                self._train_model(dataset['dataloader'], dataset['epochs'])   # errolr line

    def prediction(self, bio=True, summary=True):
        self.model_w.model.eval()
        start = time.time()
        iterator = self.data_handler.stream_files()
        for out_path, data_loader, text in iterator:
            if not self.data_handler.multi_task_mapping:
                ids, predictions, input_masks = [], [], []
            else:
                predictions = {}
                for k in self.data_handler.encoding['tag2idx'].keys():
                    predictions[k] = []
                input_masks, ids = [], []
            for batch in data_loader:
                batch = {k: (t.to(self.device) if t is not None else None) for k, t in batch.items()}
                with torch.no_grad():
                    if self.model_w.optim_grouped_params is None:
                        tag_seq, score, input_mask = self.model_w.model(char_sentence=batch['chars'], sentence=batch['ids'], lengths=batch['lengths'], feature_sentence=batch['features'].float())
                    else:
                        if not self.data_handler.multi_task_mapping:
                            outputs = self.model_w.model(batch['ids'], token_type_ids=None, attention_mask=batch['masks'], labels=batch['tags'])
                            logits = outputs[1]
                            tag_seq = torch.argmax(logits, axis=2)
                            predictions.extend(tag_seq.tolist())
                        else:
                            outputs = self.model_w.model(
                                batch['ids'], 
                                token_type_ids=None, 
                                sequence_lengths=batch['lengths'],
                                attention_mask=batch['masks'])
                            if len(self.data_handler.encoding['tag2idx']) == 4:
                                logits = {
                                    'software': outputs[1],
                                    'soft_type': outputs[2],
                                    'mention_type': outputs[3],
                                    'soft_purpose': outputs[4]
                                }
                            elif len(self.data_handler.encoding['tag2idx']) == 3:
                                logits = {
                                    'software': outputs[1],
                                    'soft_type': outputs[2],
                                    'soft_purpose': outputs[3]
                                }
                            for k, v in logits.items():
                                if self.model_w.model_type in ['MultiSciBERTCRF', 'MultiOpt2SciBERTCRF']:
                                    predictions[k].extend(v.tolist())
                                else:
                                    predictions[k].extend(torch.argmax(v, axis=2).tolist())
                
                input_mask = (
                    (batch['ids'] != self.data_handler.special_toks['cls_tok']) &
                    (batch['ids'] != self.data_handler.special_toks['pad_tok']) &
                    (batch['ids'] != self.data_handler.special_toks['sep_tok'])
                )
                ids.extend(batch['ids'].tolist())
                input_masks.extend(input_mask.tolist())

            if not self.data_handler.multi_task_mapping:
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
            else:
                pred_tags = {}
                for k in self.data_handler.encoding['tag2idx'].keys():
                    pred_tags[k] = []
                n_text = []
                for top_idx, (j, j_m) in enumerate(zip(ids, input_masks)):
                    for k in pred_tags.keys():
                        pred_tags[k].append([])
                    n_text.append([])
                    for bottom_idx, (i, i_m) in enumerate(zip(j, j_m)):
                        if i_m:
                            n_text[-1].append(i)
                            for k in pred_tags.keys():
                                pred_tags[k][-1].append(self.data_handler.encoding['tag2name'][k][predictions[k][top_idx][bottom_idx]])                            

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
