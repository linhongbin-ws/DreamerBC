model lr 0.0008 cannot converge well, even 3e-4 does not converge well


rssm deter hidden to 100 and lr to 3e-4 do not reconstruct well

opt elsilon to 1e-7 should be more stable

rssm stoch discrete to 48, still does not train well, the image does not converge

small replay buffer 8000 does not train well

image grad weight exceed 100:1 is not train goood

train_carrystate set to false give a poorer training results for world model

deter 1024 give a better results compared to deter 200

cnn+mlp seems to not as good as pure cnn