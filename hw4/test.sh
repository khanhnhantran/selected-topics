python test_v2.py \
  --checkpoint-path train_ckpt_v3/best_rainsnow_edge_v2.ckpt \
  --test-path data/hw4_realse_dataset/test \
  --output-npz ./pred_v2.npz \
  --gpu-ids 6,7