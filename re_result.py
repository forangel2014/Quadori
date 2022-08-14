import numpy as np

log_dir = './logs_ours/'
log_name = 'disease_human_20.log'

f1s = []
with open(log_dir + log_name) as f:
    lines = f.readlines()
    for line in lines:
        if "EPOCH 4" in line:
            f1s.append(float(line.split('F1-Score: ')[1].split('. (Number of Data')[0]))
            
mu = np.mean(f1s)*100
std = np.std(f1s)*100
print(mu, std)