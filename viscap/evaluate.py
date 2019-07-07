import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint


parser = argparse.ArgumentParser(
    "Evaluate and/or generate EvalAI submission file."
)
parser.add_argument(
    "--config-yml",
    default="configs/lf_disc_faster_rcnn_x101.yml",
    help="Path to a config file listing reader, model and optimization "
    "parameters.",
)
parser.add_argument(
    "--split",
    default="val",
    choices=["val", "test"],
    help="Which split to evaluate upon.",
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to VisDial v1.0 val data. This argument doesn't work when "
    "--split=test.",
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to VisDial v1.0 val dense annotations (if evaluating on val "
    "split). This argument doesn't work when --split=test.",
)
parser.add_argument(
    "--test-json",
    default="data/visdial_1.0_test.json",
    help="Path to VisDial v1.0 test data. This argument doesn't work when "
    "--split=val.",
)

parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath",
    default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint.",
)

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=-1,
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for reading data.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, atleast few tens of GBs.",
)

parser.add_argument_group("Submission related arguments")
parser.add_argument(
    "--save-ranks-path",
    default="logs/ranks.json",
    help="Path (json) to save ranks, in a EvalAI submission format.",
)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================

if args.split == "val":
    val_dataset = VisDialDataset(
        config["dataset"],
        args.val_json,
        args.val_dense_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] == "disc"
        else True,
    )
else:
    val_dataset = VisDialDataset(
        config["dataset"],
        args.test_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] == "disc"
        else True,
    )
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] == "disc"
    else 5,
    num_workers=args.cpu_workers,
)
# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], val_dataset.vocabulary)
decoder = Decoder(config["model"], val_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================

model.eval()
ranks_json = []

for _, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        output = model(batch)

    ranks = scores_to_ranks(output)
    for i in range(len(batch["img_ids"])):
        # Cast into types explicitly to ensure no errors in schema.
        # Round ids are 1-10, not 0-9
        if args.split == "test":
            ranks_json.append(
                {
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(batch["num_rounds"][i].item()),
                    "ranks": [
                        rank.item()
                        for rank in ranks[i][batch["num_rounds"][i] - 1]
                    ],
                }
            )
        else:
            for j in range(batch["num_rounds"][i]):
                ranks_json.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]],
                    }
                )

    if args.split == "val":
        sparse_metrics.observe(output, batch["ans_ind"])
        if "gt_relevance" in batch:
            output = output[
                torch.arange(output.size(0)), batch["round_id"] - 1, :
            ]
            ndcg.observe(output, batch["gt_relevance"])

if args.split == "val":
    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value}")

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))
