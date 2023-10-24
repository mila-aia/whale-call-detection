# Comet
## Setup
This page describes how to integrate `comet` into your workflow.
1. Create a Comet account
2. Get a Comet API key from your online account settings and set it as a environment variable
```
export COMET_API_KEY=YOU_API_KEY
```
3. Comet has been intgrated into Pytorch Lighting:
```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
exp_logger = CometLogger(
        project_name='whale-call-detection',
        experiment_name='your_exp_name',
        save_dir='./',
    )
trainer = Trainer(
        ....
        logger=exp_logger
    )
```
## Features
### Automatic emission tracking
Comet will then automatically use CodeCarbon (if installed) to compute the emissions of the compute source being used for model training and save that data as an asset to each experiment. A `emission.csv` file will be generated as an artefact.
### Report templates
https://www.comet.com/reports-templates/baseline-classification-models-template/reports/template/baseline-classification-model-template
### Customized panels
https://www.comet.com/docs/v2/guides/comet-dashboard/code-panels/about-panels/