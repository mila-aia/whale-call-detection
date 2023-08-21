# LSTM without hyper-parameter tuning
```
python models/lstm.py -h
usage: lstm.py [-h] [--project PROJECT] [--exp-name EXP_NAME] [--run-name RUN_NAME] [--save-dir SAVE_DIR]
               [--data-path DATA_PATH] [--input-dim INPUT_DIM] [--hidden-dim HIDDEN_DIM] [--num-layers NUM_LAYERS]
               [--num-classes NUM_CLASSES] [--bidirectional] [--reg-loss-weight REG_LOSS_WEIGHT] [--lr LR]
               [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--seed SEED] [--data-type {spec,waveform}]
               [--metric-to-optimize METRIC_TO_OPTIMIZE]

Train an LSTM model for whall call recognition.

optional arguments:
  -h, --help            show this help message and exit
  --project PROJECT     name of the project (default: whale-call-detection)
  --exp-name EXP_NAME   name of the experiment (default: test_exp)
  --run-name RUN_NAME   name of the MLflow run (default: test_run)
  --save-dir SAVE_DIR   path to the wandb logging directory (default: ./wandb_log/)
  --data-path DATA_PATH
                        string name of the data directory, containing train/val/test splits (default: ./data/)
  --input-dim INPUT_DIM
                        dimension of the input sequene to the LSTM (default: 129)
  --hidden-dim HIDDEN_DIM
                        dimension of the hidden state of the LSTM (default: 128)
  --num-layers NUM_LAYERS
                        number of layers in the LSTM (default: 3)
  --num-classes NUM_CLASSES
                        number of classes (default: 2)
  --bidirectional       whether the LSTM is bidirectional (default: False)
  --reg-loss-weight REG_LOSS_WEIGHT
                        weight for the regression loss. Note that the total loss is the sum of the cross entropy loss and the
                        regression loss (default: 0.5)
  --lr LR               learning rate (default: 0.001)
  --batch-size BATCH_SIZE
                        number of instances in a single training batch (default: 64)
  --num-epochs NUM_EPOCHS
                        the total number of epochs (default: 30)
  --seed SEED           integer value seed for global random state (default: 1234)
  --data-type {spec,waveform}
                        data type to use for training (default: spec)
  --metric-to-optimize METRIC_TO_OPTIMIZE
                        metric to optimize for model selection (default: overall_val_loss)
```

# LSTM with hyper-parameter tuning
python models/lstm-optim.py -h
```
usage: lstm-optim.py [-h] [--seed SEED] [--project PROJECT] [--exp-name EXP_NAME] [--save-dir SAVE_DIR] [--input-dim INPUT_DIM]
                     [--num-classes NUM_CLASSES] [--data-path DATA_PATH] [--data-type {spec,waveform}]
                     [--batch-size BATCH_SIZE] [--search-num-epochs SEARCH_NUM_EPOCHS] [--num-epochs NUM_EPOCHS]
                     [--optuna-db OPTUNA_DB] [--hparams-space HPARAMS_SPACE] [--n-trials N_TRIALS]
                     [--metric-to-optimize METRIC_TO_OPTIMIZE] [--optimize-direction {minimize,maximize}]

Train an LSTM model for whall call recognition.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           integer value seed for global random state (default: 1234)
  --project PROJECT     name of the project (default: whale-call-detection)
  --exp-name EXP_NAME   name of the experiment (default: test_exp)
  --save-dir SAVE_DIR   path to the wandb logging directory (default: ./wandb_log/)
  --input-dim INPUT_DIM
                        dimension of the input sequene to the LSTM (default: 129)
  --num-classes NUM_CLASSES
                        number of classes (default: 2)
  --data-path DATA_PATH
                        string name of the data directory, containing train/val/test splits (default: ./data/)
  --data-type {spec,waveform}
                        data type to use for training (default: spec)
  --batch-size BATCH_SIZE
                        number of instances in a single training batch (default: 64)
  --search-num-epochs SEARCH_NUM_EPOCHS
                        the total number of epochs for hyper-parameters search phase (default: 15)
  --num-epochs NUM_EPOCHS
                        the total number of epochs (default: 30)
  --optuna-db OPTUNA_DB
                        path to Optuna logs database (default: optuna.sqlite3)
  --hparams-space HPARAMS_SPACE
                        path to the YAML file containing the hyperparameter space (default: hyperparam.yaml)
  --n-trials N_TRIALS   number trials for the hyper-parameters search (default: 20)
  --metric-to-optimize METRIC_TO_OPTIMIZE
                        metric to optimize for model selection (default: overall_val_loss)
  --optimize-direction {minimize,maximize}
                        direction of optimization (default: minimize)
```