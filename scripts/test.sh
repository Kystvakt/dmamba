nohup python train.py dataset="PB_SubCooled_0.1" experiment="dmamba/tempvel" experiment.seed=0 experiment.exp_num=1245 > ./logs/nohup_logs/dmamba_pbsubcooled0.1_tempvel_1245.out 2>&1 &
nohup bash ./scripts/train.sh > ./logs/nohup_logs/cv.out 2>&1 &
