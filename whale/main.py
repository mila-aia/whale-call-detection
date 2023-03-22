#!/usr/bin/env python
"""
Authors
 * Ge Li and Basil Roth 2023
"""

import logging
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything
from whale.utils.callbacks import LogConfigCallback

seed_everything(42, workers=True)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point of the program."""
    LightningCLI(
        save_config_overwrite=True, save_config_callback=LogConfigCallback
    )


if __name__ == "__main__":
    main()
