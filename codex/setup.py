from setuptools import setup, find_packages, find_namespace_packages
print(find_namespace_packages())
setup(
    name="codex",
    version="2024.1.1",
    description="Coverage for Data Explorer",
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
    packages=['codex', 'modules', 'utils'],
    entry_points={
    "console_scripts": [
        "codex=codex:main"
        ]
    }
)

