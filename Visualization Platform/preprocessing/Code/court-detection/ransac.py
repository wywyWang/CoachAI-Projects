import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math

def runRANSAC(m, b, iteration, max_col):
    """
    m = [-0.06992481, -0.08753701, -0.06992481, -0.08753701]
    b = [900.90526315, 918.95919289, 902.90526315, 923.04672991]
    iteration = 100
    max_col = 1920
    """

    best_count = 0
    best_weight = None
    for i in range(iteration):
        random_x = np.random.uniform(0, max_col, 100).reshape(-1, 1)
        random_y = []
        for idx_x in range(len(random_x)):
            random_line = randint(0, len(m)-1)
            random_y += [m[random_line] * random_x[idx_x] + b[random_line]]
        
        random_y = np.array(random_y).reshape(-1, 1)
        random_x = np.hstack((random_x, np.ones(len(random_x)).reshape(-1, 1)))

        #compute regression line
        weight = np.matmul(np.linalg.inv(np.matmul(random_x.T, random_x)), np.matmul(random_x.T,random_y))

        random_x = random_x[:,0]
        inlier_threshold = 3
        inlier_count = 0
        for idx_x in range(len(random_x)):
            distance = abs(weight[0] * random_x[idx_x] - random_y[idx_x] + weight[1]) / math.sqrt(weight[0] ** 2 + 1)
            if distance <= inlier_threshold:
                inlier_count += 1

        if inlier_count > best_count:
            best_count = inlier_count
            best_weight = weight.copy()

    # #draw
    # color = ['black', 'blue', 'green', 'yellow']
    # fig = plt.figure()
    # x = np.linspace(0,max_col,100)
    # for i in range(len(m)):
    #     func = np.poly1d([m[i], b[i]])
    #     y = func(x)
    #     plt.plot(x, y, color = color[i])

    # func = np.poly1d(best_weight.flatten())
    # y = func(x)
    # plt.plot(x, y, color = 'red')

    # plt.scatter(random_x, random_y)
    # fig.savefig('test.png')

    return best_weight[0][0], best_weight[1][0]

if __name__ == "__main__":
    m = [-0.06992481, -0.08753701, -0.06992481, -0.08753701]
    b = [900.90526315, 918.95919289, 902.90526315, 923.04672991]
    iteration = 100
    max_col = 1920
    best_m, best_b = runRANSAC(m, b, iteration, max_col)

    print("best m = {}".format(best_m))
    print("best b = {}".format(best_b))