import os
import shutil
from sklearn.utils import resample

src_root = '../npz/train'
dst_root = '../npz/train_boots'

data = [i for i in range(251)]
for rs in range(100):
    print(rs)
    tra_path = os.path.join(dst_root, str(rs), 'tra')
    val_path = os.path.join(dst_root, str(rs), 'val')
    
    if not os.path.exists(tra_path):
        os.makedirs(tra_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    boot = resample(data, replace=True, n_samples=251, random_state=rs)
    oob = [x for x in data if x not in boot]
#     boot.sort()
#     dst_path = os.path.join(dst_root, str(rs))
#     if not os.path.exists(dst_path):
#         os.makedirs(dst_path)
    for i in range(251):
        shutil.copy(os.path.join(src_root, str(boot[i])+'.npz'), os.path.join(tra_path, str(i)+'.npz'))
    for j in range(len(oob)):
        shutil.copy(os.path.join(src_root, str(oob[j])+'.npz'), os.path.join(val_path, str(j)+'.npz'))