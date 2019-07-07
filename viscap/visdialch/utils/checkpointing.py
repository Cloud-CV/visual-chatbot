"""
A checkpoint manager periodically saves model and optimizer as .pth
files during training.

Checkpoint managers help with experiment reproducibility, they record
the commit SHA of your current codebase in the checkpoint saving
directory. While loading any checkpoint from other commit, they raise a
friendly warning, a signal to inspect commit diffs for potential bugs.
Moreover, they copy experiment hyper-parameters as a YAML config in
this directory.

That said, always run your experiments after committing your changes,
this doesn't account for untracked or staged, but uncommitted changes.
"""
from pathlib import Path
from subprocess import PIPE, Popen
import warnings

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    model: nn.Module
        Wrapped model, which needs to be checkpointed.
    optimizer: optim.Optimizer
        Wrapped optimizer which needs to be checkpointed.
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    step_size: int, optional (default=1)
        Period of saving checkpoints.
    last_epoch: int, optional (default=-1)
        The index of last epoch.

    Example
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt")
    >>> for epoch in range(20):
    ...     for batch in dataloader:
    ...         do_iteration(batch)
    ...     ckpt_manager.step()
    """

    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dirpath,
        step_size=1,
        last_epoch=-1,
        **kwargs,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.init_directory(**kwargs)

    def init_directory(self, config={}):
        """Initialize empty checkpoint directory and record commit SHA
        in it. Also save hyper-parameters config in this directory to
        associate checkpoints with their hyper-parameters.
        """

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        # save current git commit hash in this checkpoint directory
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")
        commit_sha_filepath = self.ckpt_dirpath / f".commit-{commit_sha}"
        commit_sha_filepath.touch()
        yaml.dump(
            config,
            open(str(self.ckpt_dirpath / "config.yml"), "w"),
            default_flow_style=False,
        )

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet. """

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if not self.last_epoch % self.step_size:
            torch.save(
                {
                    "model": self._model_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
            )

    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.

    Raises
    ------
    UserWarning
        If commit SHA do not match, or if the directory doesn't have
        the recorded commit SHA.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))

    if len(checkpoint_commit_sha) == 0:
        warnings.warn(
            "Commit SHA was not recorded while saving checkpoints."
        )
    else:
        # verify commit sha, raise warning if it doesn't match
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )

    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]
