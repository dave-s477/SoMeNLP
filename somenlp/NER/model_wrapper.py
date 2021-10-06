import torch
import torch.optim as optim
import transformers

from .models import BiLSTM_CRF, FeatureLSTM, CombinedLSTM, BERTMultiTask, BERTMultiTaskOpt2, BERTMultiTaskCRF, BERTMultiTaskOpt2CRF
from somenlp.utils import set_dropout

class ModelWrapper():
    def __init__(self, model_config, device, emb_vecs, data_handler, output_handler):
        self.config = model_config
        self.model_type = model_config['general']['type']
        self.device = device
        self.embedding = emb_vecs
        self.data_handler = data_handler
        self.output_handler = output_handler
        self.global_epoch = 1
        self.optim = None
        self.scheduler = None
        self.optim_grouped_params = None
        self.checkpoint = self.config['general']['checkpoint']
        self.current_performance = 0
        self.best_performance = 0
        self.current_is_best = True

    def _load_bi_lstm_crf(self, fct):
        char_config = self.config['model']['characters'] if 'characters' in self.config['model'] else None
        emb_config = self.config['model']['embedding'] if 'embedding' in self.config['model'] else None
        drop_config = self.config['model']['dropouts'] if 'dropouts' in self.config['model'] else None
        self.model = fct(
            self.device,
            len(self.data_handler.encoding['char2idx']), 
            len(self.data_handler.encoding['word2idx']),
            self.data_handler.encoding['tag2idx'],
            self.embedding,
            char_config,
            emb_config,
            drop_config,
            self.config['model']['gen'],
            feature_dim=self.data_handler.feature_dim
        )

    def _load_bert(self):
        self.model = transformers.BertForTokenClassification.from_pretrained(self.config['model']['pretrained']['weights'], num_labels = len(self.data_handler.encoding['tag2idx']))
        if 'dropouts' in self.config['model'] and 'all' in self.config['model']['dropouts']:
            set_dropout(self.model, self.config['model']['dropouts']['all'])

    def _load_multi_bert(self, fct):
        # get output numbers for each task
        output_nums = {}
        for k, v in self.data_handler.encoding['tag2idx'].items():
            output_nums[k] = len(v)
        self.model = fct.from_pretrained(
            self.config['model']['pretrained']['weights'], 
            num_labels = output_nums)
        if 'dropouts' in self.config['model'] and 'all' in self.config['model']['dropouts']:
            set_dropout(self.model, self.config['model']['dropouts']['all'])

    def _load_multi_bert_crf(self, fct):
        # get output numbers for each task
        output_nums = {}
        for k, v in self.data_handler.encoding['tag2idx'].items():
            output_nums[k] = len(v)
        output_nums['device'] = str(self.device)
        self.model = fct.from_pretrained(
            self.config['model']['pretrained']['weights'], 
            num_labels = output_nums)
        if 'dropouts' in self.config['model'] and 'all' in self.config['model']['dropouts']:
            set_dropout(self.model, self.config['model']['dropouts']['all'])

    def _load_checkpoint(self):
        if 'model' in self.checkpoint and self.checkpoint['model']:
            print("Loading a pre-trained checkpoint from {}".format(self.checkpoint['model']))
            if self.optim_grouped_params is None:
                self.optim = optim.Adam(self.model.parameters())
            else:
                self.optim = transformers.AdamW(self.optim_grouped_params)
            checkpoint_data = torch.load(self.checkpoint['model'], map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.optim.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.global_epoch = checkpoint_data['epoch'] + 1
            self.current_performance = checkpoint_data['current_performance']
            self.best_performance = checkpoint_data['best_performance']
        else:
            print("Training a new model")

    def _setup_fine_tuning(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        self.optim_grouped_params = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0
            }
        ]

    def save_checkpoint(self, step=0):
        torch.save({
            'epoch': self.global_epoch,
            'current_performance': self.current_performance,
            'best_performance': self.best_performance,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, '{}/ep_{}_step_{}_perf_{}.pth'.format(self.output_handler.save_dir, self.global_epoch, step, round(self.current_performance)))

    def set_optim(self, config):
        if self.optim is None or config['reset']:
            if self.optim_grouped_params is None:
                if config['lr'] > 0:
                    self.optim = optim.Adam(self.model.parameters(), lr=config['lr'])
                else:
                    self.optim = optim.Adam(self.model.parameters())
            else:
                if config['lr'] > 0:
                    self.optim = transformers.AdamW(self.optim_grouped_params, lr=config['lr'])
                else:
                    self.optim = transformers.AdamW(self.optim_grouped_params)

    def set_scheduler(self, n_steps, config):
        if config['type'] == 'constant': 
            self.scheduler = transformers.get_constant_schedule(self.optim) 
        elif config['type'] == 'constant_w_warmup': 
            self.scheduler = transformers.get_constant_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'])
        elif config['type'] == 'cosine': 
            self.scheduler = transformers.get_cosine_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'], num_training_steps=n_steps)
        elif config['type'] == 'cosine_w_restart': 
            self.scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'], num_training_steps=n_steps)
            print("Using cosine scheduler with restart - the number of restarts is not configured yet.")
        elif config['type'] == 'linear': 
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'], num_training_steps=n_steps)
        elif config['type'] == 'polynomial': 
            self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'], num_training_steps=n_steps)


        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optim, num_warmup_steps=config['n_warmup'], num_training_steps=n_steps)

    def init_model(self):
        if self.model_type == 'BiLSTM-CRF':
            model_fct = BiLSTM_CRF
            self._load_bi_lstm_crf(model_fct)
        elif self.model_type == 'FeatureLSTM':
            model_fct = FeatureLSTM
            self._load_bi_lstm_crf(model_fct)
        elif self.model_type == 'CombinedLSTM':
            model_fct = CombinedLSTM
            self._load_bi_lstm_crf(model_fct)
        elif self.model_type in ['SciBERT', 'BioBERT']:
            self._load_bert()
            self._setup_fine_tuning()
        elif self.model_type == 'MultiSciBERT':
            model_fct = BERTMultiTask
            self._load_multi_bert(model_fct)
            self._setup_fine_tuning()
        elif self.model_type == 'MultiOpt2SciBERT':
            model_fct = BERTMultiTaskOpt2
            self._load_multi_bert(model_fct)
            self._setup_fine_tuning()
        elif self.model_type == 'MultiSciBERTCRF':
            model_fct = BERTMultiTaskCRF
            self._load_multi_bert_crf(model_fct)
            self._setup_fine_tuning()
        elif self.model_type == 'MultiOpt2SciBERTCRF':
            model_fct = BERTMultiTaskOpt2CRF
            self._load_multi_bert_crf(model_fct)
            self._setup_fine_tuning()
        else:
            raise(RuntimeError("Received unsupported model type: {}".format(self.model_type)))
        self.model.to(self.device)
        if self.model_type in ['MultiSciBERTCRF', 'MultiOpt2SciBERTCRF']:
            self.model.update_device()
        self._load_checkpoint()
