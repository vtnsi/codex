from setuptools import setup

setup(
    name="codex",
    version="2024.1.0",
    install_requires=[
        "pandas>=2.1.0",
        "numpy==1.23.0",
        "Bottleneck == 1.4.0",
        "matplotlib>=3.5.1",
        "seaborn>=0.13.0",
        "statsmodels==0.14.2",
        "Pillow",
        "tqdm",
    ],
)
