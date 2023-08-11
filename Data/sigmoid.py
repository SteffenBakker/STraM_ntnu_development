import numpy as np

# Sigmoid function to interpolate
# Q: What is the name of the function?
def sigmoid(t, start_value, end_value, t_start, t_end, mid, a, k):
    # check for errors in input
    if(not (t >= t_start and t <= t_end and mid >= 0 and a > 0 and a < 1 and k > 0 and k < 1)):
        raise Exception("Incorrect sigmoid input")
    sig_fraction = 1 + (0 - 1)*((1/(1+np.exp(-k*(t - t_start - mid)))**a))  # starts at 1, goes to zero
    return sig_fraction*start_value + (1-sig_fraction)*end_value


"""
# testing:
t_start = 2023
t_end = 2050
mid = 12
a = 0.9
k = 0.4

start_value = 0
end_value = 100

for t in np.arange(t_start, t_end):
    print(sigmoid(t, start_value, end_value, t_start, t_end, mid, a, k))
"""