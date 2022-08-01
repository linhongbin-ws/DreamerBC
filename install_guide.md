```sh
conda create -n dreamerv2 python=3.7
conda activate dreamerv2
conda install cudatoolkit=11.3 -c pytorch
pip install tensorflow==2.9.0
conda install cudnn=8.2 -c anaconda
pip install protobuf==3.20.1
pip install -e .
```


python dreamerv2/train.py --logdir ./log/dmc_walker_walk/dreamerv2/1 \
  --configs dmc_vision --task dmc_walker_walk