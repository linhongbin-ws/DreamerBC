pip install gym==0.25.2

python dreamerv2/train.py --logdir ./data/minigrid/AC --configs minigrid --task minigrid_MiniGrid-DoorKey-6x6-v0
python dreamerv2/train.py --logdir ./data/minigrid/MTCS --configs minigrid mcts --task minigrid_MiniGrid-DoorKey-6x6-v0