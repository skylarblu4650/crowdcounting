# Copyright (c) 2023 H2O.ai. Proprietary License - All Rights Reserved

-------------------------------------------------------------------------------

                H2O HYDROGEN TORCH STANDALONE PYTHON SCORING PIPELINE


                           PROPRIETARY LICENSE -

                USE OF THIS SCORING PIPELINE REQUIRES A VALID,
                ONGOING LICENSE AGREEMENT WITH H2O.AI.

                Please contact sales@h2o.ai for more information.

-------------------------------------------------------------------------------

This package contains an exported model trained in H2O Hydrogen Torch and additional 
resources to predict new data and productionize.

REQUIREMENTS:
* Docker
* NVIDIA drivers version 11.4+ for GPU scoring

SETUP:

* Build a docker image: `docker build -t ht_scoring -f Dockerfile.scoring .`
* Run a docker container:
    - CPU: `docker run -it --rm --shm-size 2G ht_scoring bash`
    - GPU: `docker run -it --rm --gpus all --shm-size 2G ht_scoring bash`
* Inside docker container:
    - Run scoring pipeline: `python3.8 scoring_pipeline.py`
    - Run ONNX scoring pipeline: `python3.8 onnx_scoring.py` (if available)

DIRECTORY LISTING:

    Dockerfile.scoring      A Dockerfile that contains all the commands to assemble
                            a Hydrogen Torch scoring image.

    hydrogen_torch-*.whl    Wheel package containing the necessary hydrogen_torch
                            framework functionality for prediction.
    
    scoring_pipeline.py     An example Python script demonstrating how to load
                            the model and score new data.

    onnx_scoring.py         An example Python script demonstrating how to load
                            the ONNX model and score new data (Optional).

    checkpoint.pth          Checkpoint of trained model.

    model.onnx              Checkpoint of trained model in ONNX format (Optional).

    cfg.p                   Internal Hydrogen Torch config file.

    images                  Optional folder containing sample images from validation.

    audios                  Optional folder containing sample audios from validation.

    texts                   Optional folder containing sample texts from validation.

