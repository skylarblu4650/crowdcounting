global result
global number
from datetime import datetime
import os
import glob
import json
import dill
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from hydrogen_torch.src.models.convert_3d import convert_3d
from hydrogen_torch.src.utils.data_utils import get_inference_batch_size
from hydrogen_torch.src.utils.modeling_utils import load_checkpoint, run_python_scoring_inference

def delete_old_files(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Check the number of files
    if len(files) <= 1:
        print(f"There are {len(files)} file(s) in the folder. No files will be deleted.")
        return

    # Get the creation time of each file
    file_times = [(f, os.path.getctime(os.path.join(folder_path, f))) for f in files]

    # Sort files by creation time in descending order
    file_times.sort(key=lambda x: x[1], reverse=True)

    # Keep the latest file and delete the rest
    for file, _ in file_times[1:]:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

    print(f"Kept the latest file: {os.path.join(folder_path, file_times[0][0])}")

def run_scoring_pipeline():
    # Define the folder path
    folder_path = "/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/images"

    # Keep only the latest file
    delete_old_files(folder_path)

    # Reading the config from trained experiment
    with open("cfg.p", "rb") as pickle_file:
        cfg = dill.load(pickle_file)

    # Changing internal cfg settings for inference, not subject to change
    cfg.prediction._calculate_test_metric = False

    # Preparing exemplary dataframe for inference loading samples
    # This has to be altered for custom data

    # Image data -------------------------------------------------------
    if hasattr(cfg.dataset, "image_column"):
        images = []
        for image in sorted(glob.glob("images/*")):
            images.append(os.path.basename(image))

        test_df = pd.DataFrame({f"{cfg.dataset.image_column}": images})

        # Set image folder
        cfg.dataset.data_folder_test = "images"
    # -----------------------------------------------------------------
    #     test_df = pd.DataFrame.from_dict(questions_and_contexts)
    # # ------------------------------------------------------------------

    # Set device for inference
    if torch.cuda.is_available():
        cfg.environment._device = "cuda"
    else:
        cfg.environment._device = "cpu"

    # Disable original pretrained weights for model initialization
    if hasattr(cfg.architecture, "pretrained"):
        cfg.architecture.pretrained = False

    # It is possible to specify a custom cache directory for Huggingface models
    if hasattr(cfg, "transformers_cache_directory"):
        cfg.transformers_cache_directory = None

    # Loading model and checkpoint
    model = cfg.architecture.model_class(cfg).eval().to(cfg.environment._device)

    # Convert to 3D CNNs if needed
    if hasattr(cfg.architecture, "is_3d") and cfg.architecture.is_3d:
        model = convert_3d(model)

    cfg.architecture.pretrained_weights = "checkpoint.pth"
    load_checkpoint(cfg, model)

    # Preparing torch dataset and dataloader
    # Batch_size and num_workers are subject to change
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

    # Running actual inference
    # Raw_predictions is a dictionary with predictions in the raw format
    # df_predictions is a pandas DataFrame with predictions
    raw_predictions, df_predictions = run_python_scoring_inference(
        cfg=cfg, model=model, dataloader=test_dataloader
    )

    # Final output
    print(df_predictions.head())
    result = df_predictions['pred_label'].iloc[0]
    print(result)
    return  result

# Run the scoring pipeline
number = run_scoring_pipeline()
