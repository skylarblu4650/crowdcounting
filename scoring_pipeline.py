# Copyright (c) 2023 H2O.ai. Proprietary License - All Rights Reserved

"""Scoring pipeline for models trained in H2O Hydrogen Torch."""

import glob
import json
import os

import dill
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from hydrogen_torch.src.models.convert_3d import convert_3d
from hydrogen_torch.src.utils.data_utils import get_inference_batch_size
from hydrogen_torch.src.utils.modeling_utils import (
    load_checkpoint,
    run_python_scoring_inference,
)

if __name__ == "__main__":

    # reading the config from trained experiment
    with open("cfg.p", "rb") as pickle_file:
        cfg = dill.load(pickle_file)

    # changing internal cfg settings for inference, not subject to change
    cfg.prediction._calculate_test_metric = False

    # preparing exemplary dataframe for inference loading samples
    # this has to be altered for custom data

    # Image data -------------------------------------------------------
    if hasattr(cfg.dataset, "image_column"):
        images = []
        for image in sorted(glob.glob("images2/*")):
            images.append(os.path.basename(image))

        test_df = pd.DataFrame({f"{cfg.dataset.image_column}": images})

        # set image folder
        cfg.dataset.data_folder_test = "images2"
    # ------------------------------------------------------------------

    # Audio data -------------------------------------------------------
    if hasattr(cfg.dataset, "audio_column"):
        audios = []
        for audio in sorted(glob.glob("audios/*")):
            audios.append(os.path.basename(audio))

        test_df = pd.DataFrame({f"{cfg.dataset.audio_column}": audios})

        # set audio folder
        cfg.dataset.data_folder_test = "audios"
    # ------------------------------------------------------------------

    # Text data --------------------------------------------------------
    if hasattr(cfg.dataset, "text_column"):
        all_files = sorted(glob.glob("texts/*"))
        col_names = cfg.dataset.text_column
        if type(col_names) is str:
            col_names = [col_names]

        test_df = pd.concat(
            [pd.read_csv(x, names=col_names, dtype=str) for x in all_files]
        )
        test_df = test_df.reset_index(drop=True)

    # special handling for span prediction problem type
    if all(
        hasattr(cfg.dataset, column) for column in ("question_column", "context_column")
    ):
        questions_and_contexts = []

        for text in sorted(glob.glob("texts/*")):
            data = json.load(open(text))

            questions_and_contexts.append(
                {
                    cfg.dataset.question_column: data["question"],
                    cfg.dataset.context_column: data["context"],
                }
            )

        test_df = pd.DataFrame.from_dict(questions_and_contexts)
    # ------------------------------------------------------------------

    # set device for inference
    if torch.cuda.is_available():
        cfg.environment._device = "cuda"
    else:
        cfg.environment._device = "cpu"

    # disable original pretrained weights for model initialization
    if hasattr(cfg.architecture, "pretrained"):
        cfg.architecture.pretrained = False

    # it is possible to specify a custom cache directory for Huggingface models
    if hasattr(cfg, "transformers_cache_directory"):
        cfg.transformers_cache_directory = None

    # loading model and checkpoint
    model = cfg.architecture.model_class(cfg).eval().to(cfg.environment._device)

    # convert to 3D CNNs if needed
    if hasattr(cfg.architecture, "is_3d") and cfg.architecture.is_3d:
        model = convert_3d(model)

    cfg.architecture.pretrained_weights = "checkpoint.pth"
    load_checkpoint(cfg, model)

    # preparing torch dataset and dataloader
    # batch_size and num_workers are subject to change
    batch_size = get_inference_batch_size(cfg)

    test_dataset = cfg.dataset.dataset_class(df=test_df, cfg=cfg, mode="test")
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_df),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=test_dataset.get_validation_collate_fn(),
    )

    # running actual inference
    # raw_predictions is a dictionary with predictions in the raw format
    # df_predictions is a pandas DataFrame with predictions
    raw_predictions, df_predictions = run_python_scoring_inference(
        cfg=cfg, model=model, dataloader=test_dataloader
    )

    # final output
    print(df_predictions.head())
