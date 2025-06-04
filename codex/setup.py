from setuptools import setup, find_packages, find_namespace_packages

print(find_namespace_packages())
setup(
    name="codex",
    version="2024.1.1",
    description="Coverage for Data Explorer",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.1",
        "matplotlib==3.5.1",
        "seaborn",
        "statsmodels==0.14.2",
        "tqdm",
        "directory_tree==1.0.0",
    ],
    packages=["codex", "modules", "utils", "vis", "modes"],
    entry_points={"console_scripts": ["codex=codex:main"]},
)
