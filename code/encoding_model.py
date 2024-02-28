# =============================================================================
# Train the encoding model 
# =============================================================================
import os
import pickle
from func import train_model_THINGS2

print(f'>>> Train the encoding model <<<')
reg = train_model_THINGS2()

# Save the model
save_dir = os.path.join('dataset','THINGS_EEG2')
pickle.dump(reg, open(os.path.join(save_dir, 'reg_model.pkl'), 'wb'))