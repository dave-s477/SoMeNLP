import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CRF(nn.Module):
    
    def __init__(self, tagset_size, device, init_parameters=None):
        super(CRF, self).__init__()
        self.device = device
        if init_parameters is None:
            #self.transition_params = nn.Parameter(torch.randn(tagset_size, tagset_size))
            self.transition_params = nn.Parameter(nn.init.xavier_normal_(torch.empty(tagset_size, tagset_size)))
        else:
            self.transition_params = nn.Parameter(init_parameters.to(self.device))
        
    def crf_sequence_score(self, inputs, tag_indices, sequence_lengths):
        """Computes the unnormalized score for a tag sequence.

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
              we compute the unnormalized score.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
          sequence_scores: A [batch_size] vector of unnormalized sequence scores.
        """
        
        # If max_seq_len is 1, we skip the score calculation and simply gather the
        # unary potentials of the single tag.
        def _single_seq_fn():
            batch_size = inputs.shape[0]
            example_inds = torch.arange(batch_size, dtype=tag_indices.dtype).to(self.device).unsqueeze(1)
            sequence_scores = inputs.squeeze(1)[example_inds, tag_indices].squeeze(1)
            sequence_scores = torch.where(sequence_lengths <= 0, torch.zeros(sequence_scores.shape).to(self.device), sequence_scores)
            return sequence_scores
        
        # Compute the scores of the given tag sequence.
        def _multi_seq_fn():
            unary_scores = self.crf_unary_score(tag_indices, sequence_lengths, inputs)
            binary_scores = self.crf_binary_score(tag_indices, sequence_lengths)
            sequence_scores = unary_scores + binary_scores
            return sequence_scores

        if inputs.shape[1] == 1:
            return _single_seq_fn()
        else:
            return _multi_seq_fn()
        
    def crf_forward(self, inputs, state, sequence_lengths):
        """Computes the alpha values in a linear-chain CRF.

        See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
             values.
          sequence_lengths: A [batch_size] vector of true sequence lengths.

        Returns:
          new_alphas: A [batch_size, num_tags] matrix containing the
              new alpha values.
        """
        
        batch_size = inputs.shape[0]
        sequence_lengths, _ = torch.max(
            torch.stack(
                (torch.zeros(sequence_lengths.shape, dtype=sequence_lengths.dtype).to(self.device), sequence_lengths - 2),
                dim=1),
            dim=1)
        inputs = inputs.permute(1, 0, 2)
        transition_params_unsq = self.transition_params.unsqueeze(0)
        
        all_alphas = []
        for idx in range(inputs.shape[0]):
            state = state.unsqueeze(2)
            transition_scores = state + transition_params_unsq
            new_alphas = inputs[idx] + torch.logsumexp(transition_scores, dim=1)
            state = new_alphas
            all_alphas.append(new_alphas)
            
        all_alphas = torch.stack(all_alphas, dim=1)
        return all_alphas[torch.arange(sequence_lengths.shape[0]).to(self.device), sequence_lengths]
    
    def crf_log_norm(self, inputs, sequence_lengths):
        """Computes the normalization for a CRF.

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
          log_norm: A [batch_size] vector of normalizers for a CRF.
        """
        
        # Split up the first and rest of the inputs in preparation for the forward
        # algorithm.
        first_input = inputs.narrow(1, 0, 1).squeeze(1)
        
        # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
        # the "initial state" (the unary potentials).
        def _single_seq_fn():
            log_norm = torch.logsumexp(first_input, dim=1)
            log_norm = torch.where(sequence_lengths <= 0, torch.zeros(log_norm.shape).to(self.device), log_norm)
            return log_norm

        def _multi_seq_fn():
            rest_of_input = inputs.narrow(1, 1, inputs.shape[1]-1)
            alphas = self.crf_forward(rest_of_input, first_input, sequence_lengths)
            log_norm = torch.logsumexp(alphas, dim=1)
            log_norm = torch.where(sequence_lengths <= 0, torch.zeros(log_norm.shape).to(self.device), log_norm)
            return log_norm

        if inputs.shape[1] == 1:
            return _single_seq_fn()
        else:
            return _multi_seq_fn()
        
    def crf_log_likelihood(self, inputs, tag_indices, sequence_lengths):
        """Computes the log-likelihood of tag sequences in a CRF.

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
              we compute the log-likelihood.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
          log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
            each example, given the sequence of tag indices.
        """
        
        # Get shape information.
        num_tags = inputs.shape[2]
        
        sequence_scores = self.crf_sequence_score(inputs, tag_indices, sequence_lengths)
        log_norm = self.crf_log_norm(inputs, sequence_lengths)
        
        log_likelihood = sequence_scores - log_norm
        return log_likelihood
    
    def crf_unary_score(self, tag_indices, sequence_lengths, inputs):
        """Computes the unary scores of tag sequences.

        Args:
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
        Returns:
          unary_scores: A [batch_size] vector of unary scores.
        """
        batch_size = inputs.shape[0]
        max_seq_len = inputs.shape[1]
        num_tags = inputs.shape[2]

        flattened_inputs = torch.flatten(inputs)
        offsets = torch.unsqueeze(torch.arange(batch_size).to(self.device) * max_seq_len * num_tags, 1)
        offsets = torch.add(offsets, torch.unsqueeze(torch.arange(max_seq_len).to(self.device) * num_tags, 0))
        flattened_tag_indices = offsets + tag_indices
        flattened_tag_indices = torch.flatten(flattened_tag_indices)
        unary_scores = torch.gather(flattened_inputs, 0, flattened_tag_indices).view(batch_size, max_seq_len)
        masks = torch.arange(unary_scores.shape[1]).to(self.device)[None, :] < sequence_lengths[:, None]
        masks = masks.squeeze(1).long()
        unary_scores = torch.sum(unary_scores * masks, dim=1)
        return unary_scores
    
    def crf_binary_score(self, tag_indices, sequence_lengths):
        """Computes the binary scores of tag sequences.

        Args:
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
          binary_scores: A [batch_size] vector of binary scores.
        """
        # Get shape information.
        batch_size = tag_indices.shape[0]
        num_tags = self.transition_params.shape[0]
        num_transitions = tag_indices.shape[1] - 1
        
        start_tag_indices = tag_indices.narrow(1, 0, num_transitions)
        end_tag_indices = tag_indices.narrow(1, 1, num_transitions)
        
        # Encode the indices in a flattened representation.
        flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
        flattened_transition_indices = flattened_transition_indices.flatten()
        flattened_transition_params = self.transition_params.flatten()

        # Get the binary scores based on the flattened representation.
        binary_scores = torch.gather(flattened_transition_params, 0, flattened_transition_indices).view(batch_size, -1)

        masks = torch.arange(tag_indices.shape[1]).to(self.device)[None, :] < sequence_lengths[:, None]
        masks = masks.squeeze(1).long()
        truncated_masks = masks.narrow(1, 1, masks.shape[1]-1)
        binary_scores = torch.sum(binary_scores * truncated_masks, dim=1)
        
        return binary_scores
    
    def viterbi_decode_batch(self, feats, sequence_lengths):
        """Computes the most likely sequence with the viterbi algorithm for batch inputs.
        

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
        Returns:
          viterbi_sequence: A [batch_size, max_seq_len] vector of most likely sequences.
          viterbi_score: The corresponding score of the viterbi_sequence
          bool_mask: Input mask corresponding to the given lengths
        """
        bool_masks = torch.arange(feats.shape[1]).to(self.device)[None, :] < sequence_lengths[:, None]
        bool_masks = bool_masks.squeeze(1)

        trellis = torch.zeros(feats.shape).to(self.device)
        backpointers = torch.zeros(feats.shape, dtype=torch.int64).to(self.device)
        trellis[:, 0] = feats[:, 0]
        for t in range(1, feats.shape[1]):
            v = torch.unsqueeze(trellis[:, t - 1], 2) + self.transition_params
            val_max, arg_max = torch.max(v, dim=1)
            bool_mask = bool_masks[:, t]
            trellis[:, t] = torch.where(bool_mask.unsqueeze(1), feats[:, t] + val_max, trellis[:, t-1])
            backpointers[:, t]= torch.where(
                bool_mask.unsqueeze(1), 
                arg_max,
                torch.argmax(
                    trellis[:, t-1], 
                    dim=1).unsqueeze(1).expand(
                                    arg_max.shape))

        viterbi_cand_val, viterbi_cand_idx = torch.max(trellis[:, -1], dim=1)
        viterbi = [viterbi_cand_idx]
        reverse_bps = torch.flip(backpointers[:, 1:], dims=[1])
        for idx in range(reverse_bps.shape[1]):
            next_viterbi = torch.gather(
                    reverse_bps[:, idx], 
                    1, 
                    viterbi[-1].unsqueeze(1)
                ).squeeze(1)
            viterbi.append(next_viterbi)

        viterbi.reverse()
        viterbi_score, _ = torch.max(trellis[:, -1], dim=1)
        return torch.stack(viterbi, dim=1), viterbi_score, bool_masks