import numpy as np

def convolve(x,w):
    n_x = len(x)
    n_w = len(w)
    
    w = w[::-1]
    x = x[::-1]
    y = np.zeros(len(x))
    
    
    for i in range(1,n_x+1):
        if (i < n_w):
            d = x[0:i]
            y[(i-1)] = np.dot(d,w[-i:])
        else:
            d = x[(i-n_w):i]
            y[(i-1)] = np.dot(d,w)
            
    y = y[::-1]
    return y





def adaptivefiltering(x, w, step, n_iter):
# =============================================================================
# INICIALIZATION    
# =============================================================================
    n_x = len(x)
    n_w = len(w)
    w = w[::-1]
    x = x[::-1]
    y = np.zeros(len(x))
    
    
    
        
# Convolving process
    for i in range(1,n_x+1):
        if (i < n_w):
            d = x[0:i]
            y[(i-1)] = np.dot(d,w[-i:])
        else:
            d = x[(i-n_w):i]
            y[(i-1)] = np.dot(d,w)
    
    return y













# =============================================================================
# 
# for i in range(0,n_x):
#     if (n_x - i > n_w):
#         d = x[0+i : n_w+i]
#         y[i] = sum(d * w)
#         
#     else:
#         d = x[i:]
#         y[i] = sum(d*w[:n_x-i])
# =============================================================================




# =============================================================================
# for i in range(0,n_x - n_w):
#     d = x[0+i : n_w+i]
#     y[i] = sum(d * w)
# =============================================================================