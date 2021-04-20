import torch
import torch.nn as nn

from .crf import CRF

class CombinedLSTM(nn.Module):

    def __init__(self, device, char_size, word_size, tag2idx, emb_vecs, char_config, emb_config, drop_config, gen_config, **kwargs):
        super(CombinedLSTM, self).__init__()
        self.device = device
        self.char_size = char_size
        self.word_size = word_size
        self.tag2idx = tag2idx
        self.tag_size = len(tag2idx)
        self.char_config = char_config
        self.emb_config = emb_config
        self.drop_config = drop_config
        self.gen_config = gen_config
        if not 'feature_dim' in kwargs:
            raise(RuntimeError("FeatureLSTM requires feature_dim to be given."))
        self.feature_dim = kwargs['feature_dim'] 
        self.feature_projection = self.gen_config['feature_emb_dim'] > 0
        
        self.char_embeds = nn.Embedding(self.char_size, self.char_config['emb_dim'])
        self.char_lstm = nn.LSTM(self.char_config['emb_dim'], self.char_config['hidden_dim'], num_layers=self.char_config['layers'], bidirectional=True)

        self.word_embeds = nn.Embedding.from_pretrained(emb_vecs)
        if not self.emb_config['train']:
            self.word_embeds.requires_grad = False
        
        if self.feature_projection:
            self.feature_remap = nn.Linear(self.feature_dim, self.gen_config['feature_emb_dim'])
            feature_out_dim = self.gen_config['feature_emb_dim']
        else:
            feature_out_dim = self.feature_dim

        self.lstm = nn.ModuleList()
        for i in range(self.gen_config['layers']):
            input_size = ( self.emb_config['dim'] + (self.char_config['hidden_dim'] * 2) + feature_out_dim ) if i == 0 else self.gen_config['hidden_dim'] * 2
            self.lstm.append(nn.LSTM(input_size, self.gen_config['hidden_dim'], num_layers=1, bidirectional=True))

        self.hidden2tag = nn.Linear(self.gen_config['hidden_dim'] * 2 * self.gen_config['layers'], self.tag_size)

        self.crf = CRF(self.tag_size, device, init_parameters=None)
        
        self.drop_char_emb = nn.Dropout(self.drop_config['char_emb'])
        self.drop_char_feat = nn.Dropout(self.drop_config['char_feat'])
        self.drop_word_emb = nn.Dropout(self.drop_config['word_emb'])
        self.drop_lstm_feat = nn.Dropout(self.drop_config['lstm_feat'])
        self.drop_dense_feat = nn.Dropout(self.drop_config['dense_feat'])
        self.drop_features_in = nn.Dropout(self.drop_config['in_feat'])
        self.drop_features_proj = nn.Dropout(self.drop_config['proj_feat'])
        
        if self.gen_config['zero_init']:
            self.hidden = self._init_hidden_zero()
            self.char_hidden = self._init_hidden_zero()
        else:
            self.hidden = self._init_hidden_xavier_norm()
            self.char_hidden = self._init_hidden_xavier_norm()
        # init hidden idea: 
        # create param -> copy over batch -> concat with input length -> input to dense network 
        # -> use as initial lstm state...

    def _init_hidden_zero(self, batch_size=1, hidden_dim=100):
        """Randomly initializes the hidden state of the bi-lstm.

        Args:
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        Returns:
          (float tensor, float tensor): Random initial states for forward 
                                        and backward cells
        """
        return (torch.zeros(2, batch_size, hidden_dim).to(self.device),
                torch.zeros(2, batch_size, hidden_dim).to(self.device))
    
    def _init_hidden_xavier_norm(self, batch_size=1, hidden_dim=100):
        """Randomly initializes the hidden state of the bi-lstm.

        Args:
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        Returns:
          (float tensor, float tensor): Random initial states for forward 
                                        and backward cells
        """
        return (nn.init.xavier_normal_(torch.empty(2, batch_size, hidden_dim).to(self.device)),
                nn.init.xavier_normal_(torch.empty(2, batch_size, hidden_dim).to(self.device)))
    
    def _get_character_features(self, char_sentence):
        """Create character based features by applying a bi-lstm on a char embedding

        Args:
          char_sentence: A [batch_size, max_seq_len, max_word_len] tensor of char indices.
        Returns:
          char_features: A [batch_size, max_seq_len, feat_len] tensor of char features for each word
        """
        original_size = char_sentence.size()
        if self.gen_config['zero_init']:
            self.char_hidden = self._init_hidden_zero(original_size[0] * original_size[1], self.char_config['hidden_dim'])
        else:
            self.char_hidden = self._init_hidden_xavier_norm(original_size[0] * original_size[1], self.char_config['hidden_dim'])
        
        # Get character embedding
        char_embeds = self.char_embeds(char_sentence)
        char_embeds = self.drop_char_emb(char_embeds)
        char_embeds = char_embeds.view(original_size[0] * original_size[1], original_size[2], char_embeds.size()[-1]).permute(1, 0, 2)
        
        # Get char bi-lstm features
        char_lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
        concat_lstm_hidden = torch.cat((self.char_hidden[0][0], self.char_hidden[0][1]), dim=-1)
        concat_lstm_hidden = concat_lstm_hidden.view(original_size[0], original_size[1], -1)
        concat_lstm_hidden = self.drop_char_feat(concat_lstm_hidden)
        
        return concat_lstm_hidden
        
    def _get_lstm_features(self, sentence, char_feats, features, sequence_lengths):
        """Create word based features by applying a bi-lstm on a sentence

        Args:
          sentence: A [batch_size, max_seq_len] tensor of word indices for a word emb lookup
          char_feats: A [batch_size, max_seq_len, char_feat_len] tensor of char features.
        Returns:
          lstm_feats: A [batch_size, max_seq_len, lstm_feat_len] tensor of lstm features.
        """
        batch_size = sentence.size()[0]
                
        # Get word embeddings
        embeds = self.word_embeds(sentence)
        embeds = self.drop_word_emb(embeds)

        given_features = self.drop_features_in(features)
        if self.feature_projection:
            given_features = self.feature_remap(given_features)
            given_features = self.drop_features_proj(given_features)
        
        # Combine word and character features and mask inputs 
        concat_feats = torch.cat((embeds, char_feats, given_features), dim=-1)
        masks = torch.arange(sentence.shape[1]).to(self.device)[None, :] < sequence_lengths[:, None]
        masks = masks.squeeze(1).long().unsqueeze(2)
        concat_feats = torch.mul(concat_feats, masks).permute(1, 0, 2)
        
        # Get bi-lstm features
        lstm_outputs = []
        for i in range(self.gen_config['layers']):
            if self.gen_config['zero_init']:
                self.hidden = self._init_hidden_zero(batch_size, self.gen_config['hidden_dim'])
            else:
                self.hidden = self._init_hidden_xavier_norm(batch_size, self.gen_config['hidden_dim'])
            concat_feats, self.hidden = self.lstm[i](concat_feats, self.hidden)
            lstm_outputs.append(concat_feats.permute(1, 0, 2))

        lstm_out = torch.cat(lstm_outputs, -1)
        lstm_out = self.drop_lstm_feat(lstm_out)
        
        return lstm_out
    
    def get_features(self, char_sentence, sentence, features, lengths):
        char_feats = self._get_character_features(char_sentence)
        word_feats = self._get_lstm_features(sentence, char_feats, features, lengths)
        
        # Dense transform to num of tags
        feats = self.hidden2tag(word_feats)
        feats = self.drop_dense_feat(feats)
        return feats

    def neg_log_likelihood(self, tags, **features):
        """Calculates log_likelihood for training a bi-lstm-crf

        Args:
          char_sentence: A [batch_size, max_seq_len, max_word_len] tensor of char indices.
          sentence: A [batch_size, max_seq_len] tensor of word indices
          tags: A [batch_size, max_seq_len] tensor of ground-truth tags.
          lengths: A [batch_size, 1] tensor of actual sequence lengths for the padded input.
        Returns:
          log_likelihood: A scalar value for the log_likelihood over the entire batch
        """
        # Getting the features
        feats = self.get_features(features['char_sentence'], features['sentence'], features['feature_sentence'], features['lengths'])
        
        # Getting the CRF score
        log_likelihood = self.crf.crf_log_likelihood(feats, tags, features['lengths'].squeeze(1))
        if self.gen_config['crf_norm']:
            log_likelihood = log_likelihood / features['lengths'].squeeze(-1)
            
        tag_weights = tags != self.tag2idx['O']
        tag_weights = torch.where(tag_weights.any(1), torch.tensor(self.gen_config['sample_weight']).to(self.device), torch.tensor(1.0).to(self.device))
        log_likelihood = log_likelihood * tag_weights
        neg_log_likelihood = torch.mean(-log_likelihood)
        
        return neg_log_likelihood

    def forward(self, **features): 
        """Calculates tag sequence and its score based on the bi-lstm-crf

        Args:
          char_sentence: A [batch_size, max_seq_len, max_word_len] tensor of char indices.
          sentence: A [batch_size, max_seq_len] tensor of word indices
          lengths: A [batch_size, 1] tensor of actual sequence lengths for the padded input.
        Returns:
          log_likelihood: A scalar value for the log_likelihood over the entire batch
        """
        # Getting the features
        feats = self.get_features(features['char_sentence'], features['sentence'], features['feature_sentence'], features['lengths'])
        
        # Find the best path, given the features.
        tag_seq, seq_score, seq_mask = self.crf.viterbi_decode_batch(feats, features['lengths'].squeeze(1))
        
        return tag_seq, seq_score, seq_mask