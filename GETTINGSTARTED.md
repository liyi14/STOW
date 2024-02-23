# Getting Started with STOW

This guide will help you understand how to use the STOW implementation after you've installed it.

## Installation

Before you start, make sure you have installed STOW. If you haven't done so, see [Installation](./DOCUMENTATION/INSTALL.md)


## Configuration

The configuration files for the model are located in the `configs` directory. These files specify the configuration of the model, including its architecture, hyperparameters, and training settings. Each configuration file corresponds to a different model or training setup. By modifying these files or creating new ones, you can experiment with different model configurations and training strategies.

All configuration files in this project refer to a baseline configuration file named `baseline_config.yaml`. This baseline file contains common settings that are used across multiple models. Specific configuration files can override these baseline settings or add new ones. The `DATASETS.TRAIN` and `DATASETS.TEST` fields are used to specify the datasets for training and evaluation, respectively. The `SOLVER` field contains various training parameters, such as the batch size (`IMS_PER_BATCH`), learning rate (`BASE_LR`), and the number of training iterations (`MAX_ITER`). The `MODEL` field specifies the architecture and settings of the model. By adjusting these fields, you can control the behavior of the model and the training process.
We also provide config files, e.g. [stow_bin_multiframe.yaml](./configs/stow_configs/stow_bin_multiframe.yaml)

## Dataset

We provide a dataset for testing the implementation. This dataset should be placed in the `datasets` directory of your project. If necessary, adjust any references or paths in your code to match the location of this dataset. You can download the dataset from the following link: [Download Dataset](https://drive.google.com/drive/folders/1r4pgK3RcMtJU5Vx_eVQI_0Tj4B7vkAbK?usp=drive_link)

## Launch

To use the STOW implementation, follow these steps:

1. **Adjust the Configuration File**: Navigate to the `configs` directory and open the configuration file you want to use. Modify the settings as needed, such as the `DATASETS.TRAIN` and `DATASETS.TEST` fields for specifying the datasets, the `SOLVER` field for training parameters, and the `MODEL` field for the model architecture and settings.

2. **Adjust the `launch.json` File**: Open the `.vscode/launch.json` file in your project root. Modify the arguments in the `args` array to match your desired configuration. For example, replace `<path_to_config_file>` with the path to your configuration file, and `<dataset_to_train>` with the name of your training dataset.

3. **Launch the `launch.json` File**: In Visual Studio Code, open the Run view (`View` > `Run`), select the configuration you want to use from the dropdown menu, and then click the Run button (or press F5). This will start the training or evaluation process as specified in your `launch.json` configuration.

## Launch

To launch the implementation use a launch.json script for evaluation and for training.  It's located in the .vscode directory at the root of your project.

There are sample launch.json scripts provided: 

- *eval_bin_real*: This configuration is used to evaluate a model with a specific set of parameters. The args array contains command-line arguments that will be passed to the `train_net_frame.py `script. The `--config-file` argument specifies the configuration file to use, and the `--eval-only` argument indicates that the script should only evaluate the model, not train it.

- *train_frame_stow*: This configuration is used to train a model with a specific set of parameters. The args array contains command-line arguments that will be passed to the train_net_frame.py script. The `--config-file `argument specifies the configuration file to use.

Here is an example of the launch.json.

```json
{   
    "version": "0.2.0",
    "configurations": [
        {
            "name": "eval_bin_real",
            "type": "python",
            "request": "launch",
            "program": "train_net_frame.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--config-file",
                "./configs/stow_configs/stow_bin_multiframe.yaml",
                "--num-gpus",
                "1",
                "--eval-only",
                "MODEL.WEIGHTS",
                "./output/stow_bin_multiframe/model_final.pth>",
                "OUTPUT_DIR",
                "./output/stow_bin_multiframe",
                "DATASETS.TEST",
                "['stow_bin_real_test',]",
                "DATALOADER.NUM_WORKERS",
                "0",
                "INPUT.SAMPLING_FRAME_NUM",
                "6",
                "TEST.DETECTIONS_PER_IMAGE",
                "10",
                "MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD",
                "0.6",
                "MODEL.REID.TEST_MATCH_THRESHOLD",
                "0.2",
            ]
        },
        {
            "name": "train_frame_stow",
            "type": "python",
            "request": "launch",
            "program": "train_net_frame.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "CUDA_LAUNCH_BLOCKING": "1"
            },
            "args": [
                "--config-file",
                "<path_to_config_file>",
                "--num-gpus",
                "1",
                "SOLVER.IMS_PER_BATCH",
                "1",
                "OUTPUT_DIR",
                "./output/stow_bin_multiframe",
                "DATASETS.TRAIN",
                "['stow_bin_syn_train',]",
                "DATALOADER.NUM_WORKERS",
                "0",
            ]
        },
        // ... other configurations ...
    ]
}
```

Alternativly you can launch through the commandline: 

To train and evaluate your model, you can use the following commands or you can also use the provided shell scripts. 

**Bin Training:**

```bash
python ./train_net_frame.py --config-file ./configs/stow_configs/stow_bin_multiframe.yaml --num-gpus 1 OUTPUT_DIR ./output/stow_bin_multiframe
```
**Bin Evaluation**

```bash
python ./train_net_frame.py --config-file ./configs/stow_configs/stow_bin_multiframe.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/stow_bin_multiframe/model_final.pth OUTPUT_DIR ./output/stow_bin_multiframe INPUT.SAMPLING_FRAME_NUM 15 SOLVER.NUM_GPUS 1 MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD 0.6 MODEL.WEIGHTS ./output/stow_bin_multiframe/model_final.pth TEST.DETECTIONS_PER_IMAGE 20 MODEL.REID.TEST_MATCH_THRESHOLD 0.2
```
**Tabletop Training:**

```bash
python ./train_net_frame.py --config-file ./configs/stow_configs/stow_tabletop_multiframe.yaml --num-gpus 1 OUTPUT_DIR ./output/stow_tabletop_multiframe
```
**Tabletop Evaluation**

```bash
python ./train_net_frame.py --config-file ./configs/stow_configs/stow_tabletop_multiframe.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/stow_tabletop_multiframe/model_final.pth OUTPUT_DIR ./output/stow_tabletop_multiframe INPUT.SAMPLING_FRAME_NUM 15 SOLVER.NUM_GPUS 1 MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD 0.6 MODEL.WEIGHTS ./output/stow_tabletop_multiframe/model_final.pth TEST.DETECTIONS_PER_IMAGE 20 MODEL.REID.TEST_MATCH_THRESHOLD 0.2
```