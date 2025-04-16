import numpy as np
import os

dataset =[ 'compas','news','adult','blog']
sampler = ['dmala_','edmala_']

for d in dataset:
    for s in sampler:
        LL = []
        RMSE = []
        base_dir = './' + d
        for root, dirs, files in os.walk(base_dir):
                # Check if the current directory name contains 'edula'
            if (os.path.basename(root).startswith(s)):
                for file in files:
                        # Check if the file is a numpy file (assuming .npy format)
                    if file.endswith('ll.npy'):
                        file_path = os.path.join(root, file)
                        print(file_path)
                            # Load the numpy array
                        array = np.load(file_path)
                        LL.append(array)
                    else:
                        file_path = os.path.join(root, file)
                            # Load the numpy array
                        array = np.load(file_path)
                        RMSE.append(array)

        LL = np.concatenate(LL)
        RMSE = np.concatenate(RMSE)

        print(s, d)
        print("Mean Training LL " + str(np.mean(LL)))
        print("Std Training LL " + str(np.std(LL)))
        print("Mean Testing RMSE " + str(np.mean(RMSE)))
        print("Std Testing RMSE " + str(np.std(RMSE)))
        print("\n")

