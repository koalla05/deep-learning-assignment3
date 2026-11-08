"""Batching, the Lightning training module and prediction."""

import copy
import os
import time
from dataclasses import asdict

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader

from .decoding import decode_row
from .metrics import evaluate
from .tokenization import case_shape

MAX_WORD_CHARACTERS = 24


class Collator:
    """Pad a batch of windows; distinct tokens share one character encoding."""

    def __init__(self, char_vocabulary=None, use_case=False):
        self.char_vocabulary = char_vocabulary
        self.use_case = use_case

    def __call__(self, windows):
        longest = max(len(window.ids) for window in windows)
        size = len(windows)
        batch = dict(ids=torch.zeros((size, longest), dtype=torch.long),
                     tags=torch.full((size, longest), -100, dtype=torch.long),
                     lengths=torch.tensor([len(window.ids) for window in windows]))
        for index, window in enumerate(windows):
            count = len(window.ids)
            batch["ids"][index, :count] = torch.from_numpy(window.ids)
            batch["tags"][index, :count] = torch.from_numpy(window.tags)
        if self.use_case:
            batch["case_ids"] = torch.zeros((size, longest), dtype=torch.long)
            for index, window in enumerate(windows):
                shapes = [case_shape(surface) for surface in window.surfaces]
                batch["case_ids"][index, :len(shapes)] = torch.tensor(shapes)
        if self.char_vocabulary is not None:
            batch.update(self._characters(windows, longest))
        return batch

    def _characters(self, windows, longest):
        distinct = {"": 0}
        token_index = torch.zeros((len(windows), longest), dtype=torch.long)
        for index, window in enumerate(windows):
            for position, surface in enumerate(window.surfaces):
                token_index[index, position] = distinct.setdefault(surface, len(distinct))
        lengths = [max(1, min(len(surface), MAX_WORD_CHARACTERS)) for surface in distinct]
        char_ids = torch.zeros((len(distinct), max(lengths)), dtype=torch.long)
        for position, surface in enumerate(distinct):
            values = [self.char_vocabulary.get(c, 1) for c in surface[:MAX_WORD_CHARACTERS]]
            if values:
                char_ids[position, :len(values)] = torch.tensor(values)
        return dict(char_ids=char_ids, char_lengths=torch.tensor(lengths), token_index=token_index)


def predict_logits(model, windows, encoded, collator, device, batch_size=256):
    """One logits array per text; overlapping windows are averaged token by token."""
    model.eval()
    totals = [np.zeros((len(enc.ids), 3), dtype=np.float32) for enc in encoded]
    counts = [np.zeros(len(enc.ids), dtype=np.float32) for enc in encoded]
    order = sorted(range(len(windows)), key=lambda index: len(windows[index].ids))
    with torch.no_grad():
        for first in range(0, len(order), batch_size):
            chosen = [windows[index] for index in order[first:first + batch_size]]
            batch = {name: value.to(device) for name, value in collator(chosen).items()}
            scores = model(batch).cpu().numpy()
            for row, window in enumerate(chosen):
                end = window.start + len(window.ids)
                totals[window.item][window.start:end] += scores[row, :len(window.ids)]
                counts[window.item][window.start:end] += 1
    return [total / np.maximum(count, 1)[:, None] for total, count in zip(totals, counts)]


class TaggerModule(L.LightningModule):
    """Training logic. Inner-validation F1 drives early stopping and the best epoch."""

    def __init__(self, tagger, config, inner_rows, inner_encoded, inner_windows, collator):
        super().__init__()
        self.tagger = tagger
        self.config = config
        self.inner = (inner_rows, inner_encoded, inner_windows)
        self.collator = collator
        self.register_buffer("class_weights", torch.tensor(
            [config.weight_o, config.weight_b, config.weight_i], dtype=torch.float32))
        self.history = []
        self.best_f1 = -1.0
        self.best_state = None
        self._train_losses = []
        self._val_losses = []
        self._epoch_started = 0.0

    def on_train_epoch_start(self):
        self._epoch_started = time.monotonic()

    def training_step(self, batch, _):
        loss = self.tagger.loss(self.tagger(batch), batch, self.class_weights)
        self._train_losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=len(batch["ids"]))
        return loss

    def validation_step(self, batch, _):
        loss = self.tagger.loss(self.tagger(batch), batch, self.class_weights)
        self._val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        rows, encoded, windows = self.inner
        logits = predict_logits(self.tagger, windows, encoded, self.collator, self.device)
        crf = self.tagger.transitions()
        predictions = [decode_row(values, enc.offsets, rows[enc.row_index]["text"], crf=crf)
                       for values, enc in zip(logits, encoded)]
        ordered = [rows[enc.row_index] for enc in encoded]
        score = evaluate(ordered, predictions)["mix"]
        f1 = score["f1"]
        self.log("val_f1", f1, prog_bar=False)
        self.log("val_loss", float(np.mean(self._val_losses)), prog_bar=False)
        if self._train_losses:  # false during Lightning's internal checks
            self.history.append(dict(
                epoch=self.current_epoch + 1,
                train_loss=float(np.mean(self._train_losses)),
                val_loss=float(np.mean(self._val_losses)),
                val_f1=f1, val_precision=score["precision"], val_recall=score["recall"],
                lr=self.trainer.optimizers[0].param_groups[0]["lr"],
                seconds=time.monotonic() - self._epoch_started))
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_state = copy.deepcopy({k: v.cpu() for k, v in self.tagger.state_dict().items()})
        self._train_losses, self._val_losses = [], []

    def configure_optimizers(self):
        config = self.config
        embeddings = self.tagger.corrections or self.tagger.embedding
        embedding_parameters = list(embeddings.parameters()) if embeddings is not None else []
        embedding_ids = {id(parameter) for parameter in embedding_parameters}
        groups = [
            dict(params=[parameter for parameter in self.tagger.parameters()
                         if id(parameter) not in embedding_ids],
                 weight_decay=config.weight_decay),
            # Decay would pull corrections to pretrained vectors back towards zero.
            dict(params=embedding_parameters,
                 weight_decay=0.0 if config.pretrained != "none" else config.weight_decay),
        ]
        groups = [group for group in groups if group["params"]]
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(groups, lr=config.lr)
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(groups, lr=config.lr)
        elif config.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(groups, lr=config.lr)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(groups, lr=config.lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError(config.optimizer)
        if config.scheduler == "none":
            return optimizer
        steps = self.trainer.estimated_stepping_batches
        if config.scheduler == "onecycle":
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr,
                                                           total_steps=int(steps))
            return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=schedule, interval="step"))
        if config.scheduler == "cosine":
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))
            return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=schedule, interval="step"))
        if config.scheduler == "plateau":
            schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                                  factor=0.5, patience=0)
            return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=schedule, monitor="val_f1"))
        raise ValueError(config.scheduler)


class GradientFlow(Callback):
    """Record the mean absolute gradient per layer: the assignment's optimization health check."""

    def __init__(self, every=50):
        self.every = every
        self.records = []

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if trainer.global_step % self.every:
            return
        values = {name: float(parameter.grad.abs().mean())
                  for name, parameter in module.tagger.named_parameters()
                  if parameter.grad is not None}
        self.records.append(dict(step=trainer.global_step, gradients=values))


def make_logger(config, project="ai301-location-ner"):
    """W&B when NER_WANDB is online or offline; nothing otherwise."""
    mode = os.getenv("NER_WANDB", "off")
    if mode == "off":
        return False
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(project=project, name=config.key(), config=asdict(config),
                       offline=mode == "offline")


def fit(module, config, train_windows, inner_windows, collator, device):
    """Run training and restore the best epoch when the configuration asks for it."""
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_windows, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collator, generator=generator)
    inner_loader = DataLoader(inner_windows, batch_size=256, collate_fn=collator)
    flow = GradientFlow()
    callbacks = [flow]
    if config.patience:
        callbacks.append(EarlyStopping("val_f1", mode="max", patience=config.patience))
    trainer = L.Trainer(max_epochs=config.max_epochs, accelerator=device.type, devices=1,
                        gradient_clip_val=config.grad_clip or None, callbacks=callbacks,
                        logger=make_logger(config), enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False,
                        num_sanity_val_steps=0)
    trainer.fit(module, train_loader, inner_loader)
    if config.use_best_epoch and module.best_state is not None:
        module.tagger.load_state_dict(module.best_state)
    if trainer.logger:
        trainer.logger.experiment.finish()
    return flow.records

