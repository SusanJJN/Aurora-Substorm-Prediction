import os
import numpy as np
import torch

def load_pth(result_root):
    outputs = torch.load(os.path.join(result_root,'prediction.pth'))
    target = torch.load(os.path.join(result_root,'ground_truth.pth'))
    copy = torch.load(os.path.join(result_root,'last_frames.pth'))
    return outputs, target, copy

num = 20
save_path = '../results/'
# avg_outputs_list = []
copy_list = [np.array([]) for i in range(num)]
pred_list = [np.array([]) for i in range(num)]
for i in range(1,num+1):
    output_list = []
    for boot in range(100):
        result_root = os.path.join('./model/ConvLSTM/results/', str(boot), 'long_test', str(i))
        outputs, target, copy = load_pth(result_root)
#         print(outputs.shape)
        output_list.append(outputs.numpy())
    output_array = np.array(output_list)
    avg_outputs = np.mean(output_array,0)
    print(avg_outputs.shape)
    
#     save_path = save_root + str(i)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    np.savez(os.path.join(save_path, str(i)+'.npz'), outputs = avg_outputs)