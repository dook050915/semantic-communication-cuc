# AWGN TokenSeq2Seq L3 ablation

Status: poor-performing L3 ablation run.

Training was stopped at epoch 25/40 after validation loss kept worsening. The SNR sweep uses `checkpoint_best.pt` from epoch 9.

- Best checkpoint: epoch 9, val_loss 3.6850
- Final observed epoch: epoch 25/40, train_loss 1.4998, val_loss 4.3499
- Interpretation: treat this as a negative/failed L3 ablation sample, not as the main AWGN token baseline.
