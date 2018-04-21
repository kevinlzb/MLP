import numpy as np
x = np.arange(0.,10.,0.2)
m = len(x)
x0 = np.full(m,1.)
input_data = np.c_[x,x0]
target_data = 2 * x + 5 + np.random.rand(m)

loop_max = 10000
epsilon = 1e-3

theta = np.random.rand(2)
alpha = 0.001
diff = 0.
error = np.zeros(2)
count = 0
finish = 0
error_list = []

while count < loop_max:
    count += 1
    y_hat = np.dot(input_data,theta.T)
    dif = np.dot((y_hat-target_data).T,input_data)
    theta = theta - alpha * dif
    error_list.append(np.sum(diff) ** 2)
    if np.linalg.norm(theta - error) < epsilon:
        finish = 1
        break
    else:
        error = theta

print 'loop count = %d' % count, '\t theta:[%f,%f]' %(theta[0],theta[1])

