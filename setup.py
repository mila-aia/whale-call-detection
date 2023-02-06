#!/usr/bin/env python3
import sys
import site
import setuptools
from distutils.core import setup


# Editable install in user site directory can be allowed with this hack:
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="whale-call-detection",
    version="0.0.1",
    description="Whale Call Detection",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas==1.4.3",
        "speechbrain==0.5.13",
        "transformers==4.20.1",
        "torchaudio==0.12.1",
        "torch==1.12.1",
        "mlflow==1.27.0",
        "librosa==0.9.2",
        "pytorch_lightning==1.2.7",
        "obspy==1.3.0",
        "matplotlib==1.16.0",
    ],
    entry_points={
        "console_scripts": ["main=whale.main:main"],
    },
)
