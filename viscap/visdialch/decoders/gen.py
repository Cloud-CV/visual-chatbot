import torch
from torch import nn
from viscap.visdialch.utils.beam_search import BeamSearch
from typing import Dict, Tuple


class GenerativeDecoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary

        self.use_beam_search = config["beam_search"]

        if self.use_beam_search:
            self.beam_size = config["beam_size"]
            print(f"Using Beam Search with beam size: {self.beam_size}")
            self.beam_search = BeamSearch(
                vocabulary.EOS_INDEX,
                max_steps=20,
                beam_size=config["beam_size"]
            )

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.answer_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        # Handle forward methods by mode in config.
        self.mode = None
        if "mode" in config:
            self.mode = config["mode"]

        self.lstm_to_words = nn.Linear(
            self.config["lstm_hidden_size"], len(vocabulary)
        )

        self.dropout = nn.Dropout(p=config["dropout"])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, batch):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.

        During evaluation, assign log-likelihood scores to all answer options.

        During demo, generate sequences from an initial <SOS> token and
        encoding.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        if self.training:

            ans_in = batch["ans_in"]
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )
            ans_out = self.dropout(ans_out)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        elif self.mode is not None and self.mode == "demo":

            # batch_size, num_rounds are 1 throughout the demo loop
            batch_size, num_rounds = 1, 1
            ans_in = batch["ans_in"]

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, 1 * 1, lstm_hidden_size)
            hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            hidden = hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            cell = torch.zeros_like(hidden)

            end_token_flag = False
            max_seq_len_flag = False
            answer_indices = []

            if self.use_beam_search:
                # build the state
                state = {"hidden": hidden, "cell": cell}
                log_probabilities, all_top_k_predictions = (
                    self._beam_search(state, ans_in, batch_size, num_rounds)
                )
                # select the output sequence
                log_prob, select_indices = log_probabilities.max(-1)
                answer_indices = (
                    all_top_k_predictions[:,select_indices]
                    .view(batch_size, num_rounds, -1)
                )
                # TODO: Fix the flag conditions below
                return (True, False), answer_indices

            while end_token_flag is False and max_seq_len_flag is False:

                # shape: (1*1, 1)
                ans_in = ans_in.view(batch_size * num_rounds, -1)

                # shape: (1*1, 1, word_embedding_size)
                ans_in_embed = self.word_embed(ans_in)

                # shape: (1*1, 1, lstm_hidden_size)
                # new states are updated in (hidden, cell)
                ans_out, (hidden, cell) = self.answer_rnn(
                    ans_in_embed, (hidden, cell)
                )

                # calculate answer probabilities
                ans_scores = self.logsoftmax(self.lstm_to_words(ans_out))
                ans_probs = torch.exp(ans_scores)
                ans_probs = ans_probs.view(-1)

                # for initial time-step prevent PAD, SOS and EOS from
                # outputting, indices to remove are [PAD:0, SOS:1, EOS:2]
                if len(answer_indices) == 0:
                    ans_probs = ans_probs[3:]
                    _, ans_index = ans_probs.max(0)
                    ans_index += 3

                # for non-initial time-steps prevent PAD and SOS from
                # outputting, indices to remove are [PAD:0, SOS:1]
                else:
                    ans_probs = ans_probs[2:]
                    _, ans_index = ans_probs.max(0)
                    ans_index += 2

                answer_indices.append(ans_index)
                ans_in = ans_index

                # check flag conditions and raise
                if ans_index.item() == self.vocabulary.EOS_INDEX:
                    end_token_flag = True
                if len(answer_indices) > 20:
                    max_seq_len_flag = True

            answer_indices = torch.LongTensor(answer_indices)
            return (end_token_flag, max_seq_len_flag), answer_indices

        else:

            ans_in = batch["opt_in"]
            batch_size, num_rounds, num_options, max_sequence_length = (
                ans_in.size()
            )

            ans_in = ans_in.view(
                batch_size * num_rounds * num_options, max_sequence_length
            )

            # shape: (batch_size * num_rounds * num_options, max_sequence_length
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(
                1, batch_size * num_rounds * num_options, -1
            )
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = batch["opt_out"].view(
                batch_size * num_rounds * num_options, -1
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (
                    ans_word_scores * (target_ans_out > 0).float().cuda()
            )  # ugly

            ans_scores = torch.sum(ans_word_scores, -1)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

            return ans_scores

    def _beam_search(
            self,
            state: Dict[str, torch.Tensor],
            ans_in: torch.Tensor,
            batch_size: int,
            num_rounds: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Mention source and add comments and add arg hints
        start_preds = ans_in.view(batch_size*num_rounds,)

        # reshape: (batch_size * num_rounds, lstm_num_layers, lstm_hidden_size)
        state["hidden"] = state["hidden"].permute(1,0,2)
        state["cell"] = state["cell"].permute(1,0,2)

        # shape (all_top_k_predictions):
        # (batch_size * num_rounds, beam_size, max_length)
        # shape (log_probabilities):
        # (batch_size * num_rounds, beam_size)
        all_top_k_predictions, log_probabilities = self.beam_search.search(
            start_preds,
            state,
            self._take_step
        )

        return log_probabilities, all_top_k_predictions

    def _take_step(
            self,
            last_predictions: torch.Tensor,
            state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of
            the predictions during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state
            information needed to predict the next step, which includes the
            encoder outputs, the source mask, and the decoder hidden state
            and context. Each of these tensors has shape ``(group_size, *)``,
            where ``*`` can be any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where
            ``log_probabilities`` is a tensor of shape ``(group_size,
            num_classes)`` containing the predicted log probability of each
            class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the
            encoder outputs, source mask, and updated decoder hidden state
            and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not
            necessarily equal to ``batch_size``, since the group may contain
            multiple states for each source sentence in the batch.
        """
        # TODO: Mention source, add comments, fix line width
        # shape: (group_size, seq_len=1, word_embedding_size)
        last_predictions = last_predictions.long()
        last_predictions_embed = self.word_embed(last_predictions).unsqueeze(1)

        # reshape states to feed them into RNN: 
        # (lstm_num_layers, batch_size * num_rounds, lstm_hidden_size)
        hidden, cell = (
            state["hidden"].permute(1,0,2).contiguous(),
            state["cell"].permute(1,0,2).contiguous()
        )

        # take the rnn step
        output, (hidden, cell) = self.answer_rnn(
            last_predictions_embed,
            (hidden, cell)
        )

        # reshape and update state dict
        # shape: (batch_size * num_rounds, lstm_num_layers, lstm_hidden_size)
        state["hidden"], state["cell"] = (
            hidden.permute(1,0,2), cell.permute(1,0,2)
        )

        # compute the class log probabilities
        # shape: (group_size, num_vocab)
        output_projections = self.lstm_to_words(output).squeeze(1)
        class_log_probabilities = self.logsoftmax(output_projections)

        return class_log_probabilities, state
