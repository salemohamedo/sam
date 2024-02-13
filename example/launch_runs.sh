base_args='--use_wandb=True --wandb_tag sam_wideresnet_exps'
launch_args='/home/mila/o/omar.salemohamed/.launch -p long -m 10 -t 12'

for seed in 0 1 2; do
    for batch_size in 16 64; do
        $launch_args python train.py $base_args --seed $seed --batch_size $batch_size --rho 0. --wandb_name SGD 
        for rho in 0.05 0.1 0.5 1.; do
            $launch_args python train.py $base_args --seed $seed --batch_size $batch_size --rho $rho --sam_loss ce --wandb_name SAM_rho=$rho 
        done
    done
done

for seed in 0 1 2; do
    for batch_size in 16 64 128; do
        for rho in 0.05 0.1 0.5 1.; do
            $launch_args python train.py $base_args --seed $seed --batch_size $batch_size --rho $rho --sam_loss feat_cos_sim_pos --wandb_name SAM_feat_cos_sim_pos_rho=$rho  
        done
    done
done