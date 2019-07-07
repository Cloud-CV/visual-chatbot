import torch
from torch import nn

from visdialch.utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.option_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        # Options are variable length padded sequences, use DynamicRNN.
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, batch):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        options = batch["opt"]
        batch_size, num_rounds, num_options, max_sequence_length = (
            options.size()
        )
        options = options.view(
            batch_size * num_rounds * num_options, max_sequence_length
        )

        options_length = batch["opt_len"]
        options_length = options_length.view(
            batch_size * num_rounds * num_options
        )

        # Pick options with non-zero length (relevant for test split).
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        # shape: (batch_size * num_rounds * num_options, max_sequence_length,
        #         word_embedding_size)
        # FOR TEST SPLIT, shape: (batch_size * 1, num_options,
        #                         max_sequence_length, word_embedding_size)
        nonzero_options_embed = self.word_embed(nonzero_options)

        # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
        # FOR TEST SPLIT, shape: (batch_size * 1, num_options,
        #                         lstm_hidden_size)
        _, (nonzero_options_embed, _) = self.option_rnn(
            nonzero_options_embed, nonzero_options_length
        )

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options,
            nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device,
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        # Repeat encoder output for every option.
        # shape: (batch_size, num_rounds, num_options, max_sequence_length)
        encoder_output = encoder_output.unsqueeze(2).repeat(
            1, 1, num_options, 1
        )

        # Shape now same as `options`, can calculate dot product similarity.
        # shape: (batch_size * num_rounds * num_options, lstm_hidden_state)
        encoder_output = encoder_output.view(
            batch_size * num_rounds * num_options,
            self.config["lstm_hidden_size"],
        )

        # shape: (batch_size * num_rounds * num_options)
        scores = torch.sum(options_embed * encoder_output, 1)
        # shape: (batch_size, num_rounds, num_options)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores
