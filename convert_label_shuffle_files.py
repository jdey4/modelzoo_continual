#%%
import pickle
import numpy as np
# %%
with open('angle/model_zoo_0_1.pickle', 'rb') as f:
    res = pickle.load(f)
# %%
algo = 'model_zoo'
shifts = 6
#%%
for degree in range(0,182,4):
    res = np.zeros(2, dtype=float)
    file_to_save = 'converted_angle/'+algo+'-'+str(degree)+'.pickle'

    for shift in range(shifts):
        file_to_load = 'angle/'+algo+'_'+str(degree)+'_'+str(shift+1)+'.pickle'

        with open(file_to_load, 'rb') as f:
            tmp = np.array(pickle.load(f))
        res += 1 - tmp
    res /= shifts

    with open(file_to_save, 'wb') as f:
        pickle.dump(res, f)

# %%
