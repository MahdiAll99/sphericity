# Sphericity Radiomics Feature Extractor

This is a lightweight Python package designed for the extraction of the radiomics feature *sphericity*, and it adheres to the IBSI standards.

## Installation

### Python Installation

Ensure that Python 3.8 or a higher version is installed. If not, follow the instructions [here](https://github.com/MahdiAll99/MEDimage/blob/main/python.md).

### Package Installation

The package can be installed only from the source.

**Clone the repository:**

```
git clone https://github.com/MahdiAll99/sphericity.git
```
## Using requirements.txt

Install the required packages using the following command adn you are good to go:

```
pip install -r requirements.txt
```

## Using Anaconde

1. Navigate to the repository and create a conda environment, then install the package:

```
conda env create -f environment.yml
```

```
conda activate sphericity
```

3. To use the test notebook, add your installed environment to Jupyter Notebook kernels:

```
python -m ipykernel install --user --name=sphericity
```
---
Be aware of the prevalent issue that may arise when executing Jupyter Notebooks:
https://github.com/microsoft/azuredatastudio/issues/24436
---

## Test data

The test data can be found in the folder *`test_data`*. The data is in the NIfTI format, CT modality, and it is the [IBSI](https://theibsi.github.io/datasets/) test data for phase 2 (*mask_ct.nii.gz*).

## Example Usage

Explore the provided [notebook](https://colab.research.google.com/github/MahdiAll99/sphericity/blob/main/PackageTest.ipynb) for a detailed example on using the package.

## License
### GPL3 LICENSE SYNOPSIS

**_TL;DR_*** Here's what the license entails:

```markdown
1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

More information on about the [LICENSE can be found here](http://choosealicense.com/licenses/gpl-3.0/)
