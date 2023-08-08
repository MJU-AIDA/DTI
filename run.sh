# python train.py -d vec -e ddi_hop2 --gpu=1 --hop=2 --batch=128 -b=4 --num_epochs 100 --lr 0.00005 # trial1
python train.py -d vec -e dti_hop2 --gpu=1 --hop=2 --batch=64 -b=4 --num_epochs 100 --lr 0.00005 --feat_dim 1024 --pfeat_dim 1024

# python train.py -d vec -e ddi_hop2 --gpu=1 --hop=2 --batch=128 -b=4 -dim 64 --num_epochs 30 --lr 0.001 