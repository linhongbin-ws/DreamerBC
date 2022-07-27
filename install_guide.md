```sh
conda create -n dreamerv2 python=3.7
conda activate dreamerv2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorflow==2.9.0
conda install cudnn==8.2 -c anaconda
pip install protobuf==3.20.1
pip install -e .
```