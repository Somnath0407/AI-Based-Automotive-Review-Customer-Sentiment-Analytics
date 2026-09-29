"""
src/models/trainer.py
──────────────────────
HuggingFace Trainer-based fine-tuning pipeline for ABSA.

Wraps the HuggingFace Trainer API to fine-tune BERT or XLM-RoBERTa
on automotive review sentiment data.

Usage
-----
    from src.models.trainer import ABSATrainer
    trainer = ABSATrainer(model_key="bert")
    trainer.train(df, output_dir="models/checkpoints/bert")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config_loader import get_training_config, get_project_root
from src.models.absa_dataset import ABSADataset
from src.models.model_factory import load_tokenizer_and_model

logger = logging.getLogger(__name__)


class ABSATrainer:
    """
    Fine-tunes a transformer model for aspect-based sentiment classification.

    Parameters
    ----------
    model_key : "bert" or "xlmr"
    """

    def __init__(self, model_key: str):
        self.model_key = model_key
        self._cfg = get_training_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict:
        """
        Fine-tune the model on the provided DataFrame.

        Parameters
        ----------
        df               : DataFrame with columns Review, Aspect, Sentiment.
        output_dir       : Directory to save checkpoints. Defaults to
                           models/checkpoints/<model_key>.
        progress_callback: Optional callable(stage: str, fraction: float).

        Returns
        -------
        dict with training metrics.
        """
        try:
            from transformers import Trainer, TrainingArguments
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError as exc:
            raise ImportError(
                "transformers and scikit-learn are required for training. "
                "Install with: pip install transformers scikit-learn"
            ) from exc

        # ── Resolve output dir ────────────────────────────────────────────────
        if output_dir is None:
            output_dir = str(
                get_project_root() / "models" / "checkpoints" / self.model_key
            )
        os.makedirs(output_dir, exist_ok=True)

        if progress_callback:
            progress_callback("Preparing dataset…", 0.05)

        # ── Prepare data ──────────────────────────────────────────────────────
        df = df.dropna(subset=["Review", "Aspect", "Sentiment"]).reset_index(drop=True)

        reviews = df["Review"].tolist()
        aspects = df["Aspect"].tolist()
        labels = ABSADataset.labels_from_strings(df["Sentiment"].tolist())

        split_cfg = self._cfg.get("dataset_split", {"train": 0.70, "val": 0.15, "test": 0.15})
        val_size = split_cfg.get("val", 0.15)
        test_size = split_cfg.get("test", 0.15)

        # First split off test set
        (rv_tv, rv_test,
         as_tv, as_test,
         lb_tv, lb_test) = train_test_split(
            reviews, aspects, labels,
            test_size=test_size,
            stratify=labels,
            random_state=self._cfg.get("seed", 42),
        )

        # Then split remaining into train/val
        val_frac = val_size / (1.0 - test_size)
        (rv_train, rv_val,
         as_train, as_val,
         lb_train, lb_val) = train_test_split(
            rv_tv, as_tv, lb_tv,
            test_size=val_frac,
            stratify=lb_tv,
            random_state=self._cfg.get("seed", 42),
        )

        if progress_callback:
            progress_callback("Loading tokenizer & model…", 0.15)

        # ── Load model ────────────────────────────────────────────────────────
        tokenizer, model, device = load_tokenizer_and_model(self.model_key)
        max_length = 256

        train_dataset = ABSADataset(rv_train, as_train, tokenizer, max_length, lb_train)
        val_dataset = ABSADataset(rv_val, as_val, tokenizer, max_length, lb_val)
        test_dataset = ABSADataset(rv_test, as_test, tokenizer, max_length, lb_test)

        if progress_callback:
            progress_callback("Configuring training…", 0.20)

        # ── Training arguments ────────────────────────────────────────────────
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self._cfg.get("epochs", 5),
            per_device_train_batch_size=self._cfg.get("batch_size", 16),
            per_device_eval_batch_size=self._cfg.get("batch_size", 16),
            learning_rate=self._cfg.get("learning_rate", 2e-5),
            warmup_steps=self._cfg.get("warmup_steps", 100),
            weight_decay=self._cfg.get("weight_decay", 0.01),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            seed=self._cfg.get("seed", 42),
            report_to="none",
            no_cuda=(device.type == "cpu"),
        )

        def compute_metrics(eval_pred):
            logits, label_ids = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(label_ids, preds)
            f1 = f1_score(label_ids, preds, average="macro", zero_division=0)
            return {"accuracy": acc, "f1": f1}

        if progress_callback:
            progress_callback("Training in progress…", 0.25)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()

        if progress_callback:
            progress_callback("Evaluating on test set…", 0.90)

        # ── Save best model ───────────────────────────────────────────────────
        best_model_dir = os.path.join(output_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        logger.info("Best model saved to: %s", best_model_dir)

        # ── Test evaluation ───────────────────────────────────────────────────
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

        if progress_callback:
            progress_callback("Training complete!", 1.0)

        return {
            "train_loss": train_result.training_loss,
            "test_accuracy": test_metrics.get("test_accuracy", 0.0),
            "test_f1": test_metrics.get("test_f1", 0.0),
            "best_model_dir": best_model_dir,
            "train_samples": len(rv_train),
            "val_samples": len(rv_val),
            "test_samples": len(rv_test),
        }
