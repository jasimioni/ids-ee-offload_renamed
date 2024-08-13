# An Energy-efficient Intrusion Detection Offloading Based on
# an Early-Exit DNN for Edge-Cloud Computing



## Directory struct review

- `models`: DNN models with Early-Exits implementation
- `dataset`: sample dataset information for testing purposes
- `trained_models`: pre-trained models in pytorch model

## Project Setup

Create and setup a new virtual environment

```
python3 -mvenv venv
source venv/bin/activate
```

Install required dependencies

```
pip install -r requirements.txt
```

## Training the early exits models

```
$ dnn/train-early-exit-network.py --help
cuda
usage: train-early-exit-network.py [-h] [--glob GLOB] [--batch_size BATCH_SIZE] [--epochs EPOCHS] --dataset-folder DATASET_FOLDER [--model {alexnet,mobilenet}] [--output-folder OUTPUT_FOLDER]

options:
  -h, --help            show this help message and exit
  --glob GLOB           Glob pattern for dataset - default is 2016_01
  --batch_size BATCH_SIZE
                        Batch size for training - default is 1000
  --epochs EPOCHS       Number of epochs to train for - default is 5
  --dataset-folder DATASET_FOLDER
                        Dataset folder to get the data from
  --model {alexnet,mobilenet}
                        Model to train
  --output-folder OUTPUT_FOLDER
                        Output folder for the model - default is "saves"
```

```
dnn/train-early-exit-network.py --dataset-folder dataset --model mobilenet --epochs 1000 --output-folder /tmp
dnn/train-early-exit-network.py --dataset-folder dataset --model alexnet --epochs 1000 --output-folder /tmp
```

Every epoch results will be saved in the output folder and the most suitable one can be picked.

The trained models used in the paper can be downloaded running:

```
trained_models/download-trained-models.py
```

## Calibration

```
$ calibration/Calibrate2exits.py --help
usage: Calibrate2exits.py [-h] --trained-model TRAINED_MODEL --calibrated-model-savefile CALIBRATED_MODEL_SAVEFILE [--model MODEL] [--savefolder SAVEFOLDER] [--batch-size BATCH_SIZE]
                          [--max-iter MAX_ITER] [--epochs EPOCHS] --dataset DATASET

options:
  -h, --help            show this help message and exit
  --trained-model TRAINED_MODEL
                        .pth file to open
  --calibrated-model-savefile CALIBRATED_MODEL_SAVEFILE
                        .pth file to save
  --model MODEL         Model to choose - [alexnet | mobilenet]
  --batch-size BATCH_SIZE
                        Batch size
  --max-iter MAX_ITER   Max iterations for temperature scaling
  --epochs EPOCHS       Number of epochs for training
  --dataset DATASET     Dataset to use
```

```
calibration/Calibrate2exits.py --model alexnet --trained-model trained_models/AlexNetWithExits.pth \
                               --calibrated-model-savefile AlexNetWithExits_calibrated.pth --dataset dataset/2016_02.csv

calibration/Calibrate2exits.py --model mobilenet --trained-model trained_models/MobileNetV2WithExits.pth \
                               --calibrated-model-savefile MobileNetV2WithExits_calibrated.pth --dataset dataset/2016_02.csv
```


## NSGA2 Operation Point

### Model evaluation

The first step is to generate a csv file with the model evaluation for a sample dataset
in the format:

`y,y_exit_1,cnf_exit_1,bb_time_exit_1,exit_time_exit_1,y_exit_2,cnf_exit_2,bb_time_exit_2,exit_time_exit_2`

```
$ evaluations/generate-model-evaluation-csv.py --help
usage: generate-model-evaluation-csv.py [-h] --trained-model TRAINED_MODEL [--model MODEL] [--batch-size BATCH_SIZE] --dataset DATASET --savefile SAVEFILE

options:
  -h, --help            show this help message and exit
  --trained-model TRAINED_MODEL
                        .pth file to open
  --model MODEL         Model to choose - [alexnet | mobilenet]
  --batch-size BATCH_SIZE
                        Batch size
  --dataset DATASET     Dataset to use
  --savefile SAVEFILE   File to save to
```

```
evaluations/generate-model-evaluation-csv.py --trained-model trained_models/AlexNetWithExits_calibrated.pth --model alexnet \
                                             --dataset dataset/2016_01.csv --savefile evaluations/alexnet/2016_01_eval.csv

evaluations/generate-model-evaluation-csv.py --trained-model trained_models/MobileNetV2WithExits_calibrated.pth --model mobilenet \
                                             --dataset dataset/2016_01.csv --savefile evaluations/mobilenet/2016_01_eval.csv
```

### Multi-objective optimization (NSGA2)

```
$ nsga2/nsga2_2variables.py --help
usage: nsga2_2variables.py [-h] [--min-acceptance MIN_ACCEPTANCE] [--eval-file EVAL_FILE] [--savefile SAVEFILE] [--offspring OFFSPRING] [--gen GEN] [--population POPULATION]

options:
  -h, --help            show this help message and exit
  --min-acceptance MIN_ACCEPTANCE
                        Minimum acceptance rate (default: 0.7)
  --eval-file EVAL_FILE
                        Evaluation file pattern
  --savefile SAVEFILE   Save file name
  --offspring OFFSPRING
                        Number of offsprings (default: 80)
  --gen GEN             Number of generations (default: 1000)
  --population POPULATION
                        Population size (default: 100)
```

```
nsga2/nsga2_2variables.py --eval-file evaluations/mobilenet/2016_01_eval.csv --savefile evaluations/mobilenet/mobilenet_nsga_2016_01.bin
nsga2/nsga2_2variables.py --eval-file evaluations/alexnet/2016_01_eval.csv --savefile evaluations/alexnet/alexnet_nsga_2016_01.bin
```

### Plotting the output (and choosing the operation point)

```
$ nsga2/gen-op-point-chart.py --help
usage: gen-op-point-chart.py [-h] --datafile DATAFILE --savefile SAVEFILE [--operation-point OPERATION_POINT]

options:
  -h, --help            show this help message and exit
  --datafile DATAFILE   Path to the file to load
  --savefile SAVEFILE   Path to the file to save
  --operation-point OPERATION_POINT
                        Operation point coordinates (Array position of options - def: 5)
```

```
nsga2/gen-op-point-chart.py --datafile evaluations/saved/alexnet_x_f_0.9_2016_23.sav \
                            --savefile evaluations/op-point-alexnet.png --operation-point 55
nsga2/gen-op-point-chart.py --datafile evaluations/saved/mobilenet_x_f_0.9_2016_23.sav \
                            --savefile evaluations/op-point-mobilenet.png --operation-point 55
```

![NSGA Op Point](figs/nsga.png)

```
Quality                                              92.57036
Accuracy                                             0.074244
Time                                                 0.074349
n_1                                                  0.913411
a_1                                                  0.732164
n_2                                                  0.926571
a_2                                                   0.90211
```

Take note of *n_1*, *a_1*, *n_2*, *a_2*.

## Performing an inference

## Cloud offloading

### RabbitMQ Setup

### Remote Consumer Setup

### Inference with offloading enabled

## Evaluation Results