import torch
from torch import nn
from torch.nn import functional as F

from viscap.visdialch.utils import DynamicRNN


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # project image features to lstm_hidden_size for computing attention
        self.image_features_projection = nn.Linear(
            config["img_feature_size"], config["lstm_hidden_size"]
        )

        # fc layer for image * question to attention weights
        self.attention_proj = nn.Linear(config["lstm_hidden_size"], 1)

        # fusion layer (attended_image_features + question + history)
        fusion_size = (
            config["img_feature_size"] + config["lstm_hidden_size"] * 2
        )
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        nn.init.kaiming_uniform_(self.image_features_projection.weight)
        nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # shape: (batch_size, 10, max_sequence_length * 2 * 10)
        # concatenated qa * 10 rounds
        hist = batch["hist"]
        # num_rounds = 10, even for test (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()
        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)
        # shape: (batch_size * num_rounds, max_sequence_length,
        #         lstm_hidden_size)
        _, (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])

        # project down image features and ready for attention
        # shape: (batch_size, num_proposals, lstm_hidden_size)
        projected_image_features = self.image_features_projection(img)

        # repeat image feature vectors to be provided for every round
        # shape: (batch_size * num_rounds, num_proposals, lstm_hidden_size)
        projected_image_features = (
            projected_image_features.view(
                batch_size, 1, -1, self.config["lstm_hidden_size"]
            )
            .repeat(1, num_rounds, 1, 1)
            .view(batch_size * num_rounds, -1, self.config["lstm_hidden_size"])
        )

        # computing attention weights
        # shape: (batch_size * num_rounds, num_proposals)
        projected_ques_features = ques_embed.unsqueeze(1).repeat(
            1, img.shape[1], 1
        )
        projected_ques_image = (
            projected_ques_features * projected_image_features
        )
        projected_ques_image = self.dropout(projected_ques_image)
        image_attention_weights = self.attention_proj(
            projected_ques_image
        ).squeeze()
        image_attention_weights = F.softmax(image_attention_weights, dim=-1)

        # shape: (batch_size * num_rounds, num_proposals, img_features_size)
        img = (
            img.view(batch_size, 1, -1, self.config["img_feature_size"])
            .repeat(1, num_rounds, 1, 1)
            .view(batch_size * num_rounds, -1, self.config["img_feature_size"])
        )

        # multiply image features with their attention weights
        # shape: (batch_size * num_rounds, num_proposals, img_feature_size)
        image_attention_weights = image_attention_weights.unsqueeze(-1).repeat(
            1, 1, self.config["img_feature_size"]
        )
        # shape: (batch_size * num_rounds, img_feature_size)
        attended_image_features = (image_attention_weights * img).sum(1)
        img = attended_image_features

        # embed history
        hist = hist.view(batch_size * num_rounds, -1)
        hist_embed = self.word_embed(hist)

        # shape: (batch_size * num_rounds, lstm_hidden_size)
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])

        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        # shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding
