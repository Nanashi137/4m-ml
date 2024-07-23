### Installation

Create a new conda environment, then install the package and its dependencies:
```
conda create -n fourm python=3.9 -y
conda activate fourm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install rembg[gpu]  #install external segmentation model
```



