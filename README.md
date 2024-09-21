## 更改

# 1 /data/dataset_sr.py
在DatasetSR中读取了B0

# 2 /models/model_plain.py
在optimize_parameters函数中，更改了损失函数为(2GT - B0 - B1)

# 3 训练函数使用 main_train_psnr_loss.py
实例：python -m torch.distributed.launch --nproc_per_node=6 main_train_psnr_loss.py --epoch 10000 --opt /root/gsr/loss_change/change_loss.json  --dist True

# 4 测试函数使用 /root/gsr/KAIR/main_test_swinir_loss.py
实例：python main_test_swinir_loss.py --task classical_sr --scale 4 --training_patch_size 64 --model_path /root/gsr/loss_change/loss_change/B1_try3_patch/models/5000_G.pth --folder_lq /tmp/dataset/Set5_LR --folder_gt /tmp/dataset/Set5_GTmod12 --folder_b0 /tmp/dataset/Set5_B0_another

# 5 change_loss.json 是config文件