import sys
from utils import Logger
sys.stdout = Logger()


print('test')

sys.stdout.flush()

print('hello world')