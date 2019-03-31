import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Perceptron():
    ''' perceptron  algorithm '''

    def __init__(self):
        self.w = np.random.rand(3 , 1)
        print("initial w: {}".format(self.w))

    # reads data-set and transform w over each iteration
    def transform(self, df):

        n_epoch = 50
        for epoch in range(n_epoch):
            flag=self.modify_w(df)
            self.plotting(df, epoch)
            if flag == 0:
                print("Epoch :{}".format(epoch+1))
                break
        print("final w: {}".format(self.w))

    # modify w by calculating error over each data
    def modify_w(self, df):
        l_rate = 0.01
        flag = 0
        for fline in df:
            pred_val = self.w[0] + self.w[1] * fline[1] + self.w[2] * fline[2]
            act_res = int(fline[3])

            if pred_val >= 0:
                pred_res = 1
            else:
                pred_res = 0

            if act_res != pred_res:
                flag = 1
                e = act_res - pred_res
                self.w[0] = self.w[0] + l_rate * e
                self.w[1] = self.w[1] + l_rate * e * fline[1]
                self.w[2] = self.w[2] + l_rate * e * fline[2]
        return flag

    # plots the figure and saves it
    def plotting(self, df, epoch):

        clss = df[:, -1]

        C0 = df[clss == 0]
        C1 = df[clss == 1]
        xc1 = -4
        yc1 = (-self.w[0] - self.w[1] * xc1) / self.w[2]
        xc2 = 4
        yc2 = (-self.w[0] - self.w[1] * xc2) / self.w[2]

        xcord = []
        ycord = []

        xcord.append(xc1)
        xcord.append(xc2)
        ycord.append(yc1)
        ycord.append(yc2)

        plt.scatter(C0[:, 1], C0[:, 2], label='Class - 0', c='r')
        plt.scatter(C1[:, 1], C1[:, 2], label='Class - 1', c='b')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter Plot')
        plt.legend()
        plt.plot(xcord, ycord, c='green')
        plt.savefig('plot' + str(epoch) + '.png')
        plt.close()


classifier = Perceptron()
df = pd.read_csv("dataset_3.csv")
df = df.values
classifier.transform(df)
