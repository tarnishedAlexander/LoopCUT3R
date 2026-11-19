import json
import matplotlib.pyplot as plt
import sys
import os

log_file = 'checkpoints/dualdpt_ray_aware/log.txt'
out_file = 'training_convergence.png'

epochs = []
train_loss = []
pose_cross = []
pose_self = []
ray_loss = []
conf_loss = []

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith('{'): continue
        try:
            data = json.loads(line)
        except:
            continue
        if 'train_loss' in data:
            epochs.append(data.get('train_epoch', data.get('epoch', 0)))
            train_loss.append(data['train_loss'])
            pose_self.append(data.get('train_Regr3DPose_self_pts3d/2', 0))
            pose_cross.append(data.get('train_Regr3DPose_pts3d/2', 0))
            ray_loss.append(data.get('train_RayConsistencyLoss', 0))
            conf_loss.append(data.get('train_ConfLoss(Regr3DPose(L21Loss()))_conf_loss/2', 0))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Total Loss', color='black', linewidth=2.5)
plt.plot(epochs, pose_cross, label='Pose 3D (Cross)', linestyle='--', color='red')
plt.plot(epochs, pose_self, label='Pose 3D (Self)', linestyle='-.', color='orange')
plt.plot(epochs, ray_loss, label='Ray Consistency', color='blue')
plt.plot(epochs, conf_loss, label='Confidence Loss', color='purple', alpha=0.6)

plt.title('Recurrent DualDPT Training Convergence')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(out_file, dpi=300)
print(f"Saved {out_file}")
