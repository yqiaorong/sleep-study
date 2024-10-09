import os
# os.system('python3 code/validation/permutation_test.py --networks ResNet --layer_name fc')
# os.system('python3 code/validation/permutation_test.py --networks mpnet')

import time
from datetime import datetime 

start_time = '20:10'
current_time = datetime.now().strftime('%H:%M')

while current_time != start_time:
    print(current_time, 'wait...')
    time.sleep(60)
    current_time = datetime.now().strftime('%H:%M')
print('Start!')

os.system('python3 code/validation/permutation_test.py --networks ResNet --layer_name maxpool')
os.system('python3 code/validation/permutation_test.py --networks ResNet --layer_name layer3')