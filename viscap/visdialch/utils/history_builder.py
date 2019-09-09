from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from viscap.visdialch.data import Vocabulary


def pad_sequences(
        config: Dict,
        vocabulary: Vocabulary,
        sequences: List[List[int]]):
    """Given tokenized sequences (either questions, answers or answer
    options, tokenized in ``__getitem__``), padding them to maximum
    specified sequence length. Return as a tensor of size
    ``(*, max_sequence_length)``.

    This method is only called in ``__getitem__``, chunked out separately
    for readability.

    Parameters
    ----------
    sequences : List[List[int]]
        List of tokenized sequences, each sequence is typically a
        List[int].

    Returns
    -------
    torch.Tensor, torch.Tensor
        Tensor of sequences padded to max length, and length of sequences
        before padding.
    """

    for i in range(len(sequences)):
        sequences[i] = sequences[i][
                       : config["max_sequence_length"] - 1
                       ]
    sequence_lengths = [len(sequence) for sequence in sequences]

    # Pad all sequences to max_sequence_length.
    maxpadded_sequences = torch.full(
        (len(sequences), config["max_sequence_length"]),
        fill_value=vocabulary.PAD_INDEX,
    )
    padded_sequences = pad_sequence(
        [torch.tensor(sequence) for sequence in sequences],
        batch_first=True,
        padding_value=vocabulary.PAD_INDEX,
    )
    maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
    return maxpadded_sequences, sequence_lengths


def get_history(
        config,
        vocabulary,
        caption: List[int],
        questions: List[List[int]],
        answers: List[List[int]],
        drop_last_item: bool = True,
):
    # Allow double length of caption, equivalent to a concatenated QA pair.
    caption = caption[: config["max_sequence_length"] * 2 - 1]

    for i in range(len(questions)):
        questions[i] = questions[i][
                       : config["max_sequence_length"] - 1
                       ]

    for i in range(len(answers)):
        answers[i] = answers[i][: config["max_sequence_length"] - 1]

    # History for first round is caption, else concatenated QA pair of
    # previous round.
    history = []
    history.append(caption)
    for question, answer in zip(questions, answers):
        history.append(question + answer + [vocabulary.EOS_INDEX])
    # Drop last entry from history (there's no eleventh question).
    if drop_last_item:
        history = history[:-1]
    max_history_length = config["max_sequence_length"] * 2

    if config.get("concat_history", False):
        # Concatenated_history has similar structure as history, except it
        # contains concatenated QA pairs from previous rounds.
        concatenated_history = []
        concatenated_history.append(caption)
        for i in range(1, len(history)):
            concatenated_history.append([])
            for j in range(i + 1):
                concatenated_history[i].extend(history[j])

        max_history_length = (
                config["max_sequence_length"] * 2 * len(history)
        )
        history = concatenated_history

    history_lengths = [len(round_history) for round_history in history]
    maxpadded_history = torch.full(
        (len(history), max_history_length),
        fill_value=vocabulary.PAD_INDEX,
    )
    padded_history = pad_sequence(
        [torch.tensor(round_history) for round_history in history],
        batch_first=True,
        padding_value=vocabulary.PAD_INDEX,
    )
    maxpadded_history[:, : padded_history.size(1)] = padded_history
    return maxpadded_history, history_lengths
