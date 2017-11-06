import random
import numpy as np
import matplotlib.pyplot as plt
from data import inp

eta = 0     # learning rate
eta0 = 0    # coefficient of the learning rate

d0 = 1      # coefficient of the distance
d = d0      # distance to the nearest neighbors

t = 0       # number of repetitions (training)
w = []      # matrix of weights

wd = 10     # number of nodes at the output layer (in width)
ln = 10     # number of nodes at the output layer (in length)


def init():
    # generating random weights
    # each node at the output layer has 63 inputs from input nodes (fully connected)
    for i in range(wd):
        w.append([])
        for j in range(ln):
            w[i].append([])
            for k in range(63):
                w[i][j].append(random.uniform(0, 1))

    # set a learning rate
    global eta0, eta
    while eta0 == 0:
        try:
            eta0 = float(input('Input a coefficient of the learning rate: '))
            if eta0 <= 0 or eta0 > 1:
                print("Learning rate should be within 0 and 1!")
                eta0 = 0
            else:
                eta = eta0
        except ValueError:
            print('Not a number')

    # set a number of repetitions
    global t
    while t == 0:
        try:
            t = int(input('Input a number of repetitions: '))
            if t <= 0:
                print("Number of repetitions should be more than 0!")
                t = 0
        except ValueError:
            print('Not a number')


def train():
    print("Training...")
    iteration = 0

    while iteration != t:
        iteration += 1
        if iteration % 1000 == 0:
            print("Iteration #%d" % iteration)

        # iteration through fonts
        for f in range(len(inp)):
            # iteration through letters
            for l in range(len(inp[f])):
                # find the closest node
                minX = minY = 0
                minD = 1000000
                for x in range(len(w)):
                    for y in range(len(w[x])):
                        dist = 0
                        for i in range(len(w[x][y])):
                            dist += abs(w[x][y][i] - inp[f][l][i])

                        if dist < minD:
                            minD = dist
                            minX = x
                            minY = y

                # update weights for the closest node and its neighbors
                global d
                if d != 0:
                    for x in range(minX - d, minX + d + 1):
                        for y in range(minY - d, minY + d + 1):
                            updateWeights(x, y, f, l)
                else:
                    updateWeights(minX, minY, f, l)

        # update learning rate
        global eta
        eta = eta0 * (1 - iteration / float(t))
        # update distance to neighborhoods
        d = int(np.ceil(d0 * (1 - iteration / float(t))))


def updateWeights(x, y, f, l):
    if 0 <= x < len(w) and 0 <= y < len(w[x]):
        for i in range(len(w[x][y])):
            w[x][y][i] = w[x][y][i] + eta * (inp[f][l][i] - w[x][y][i])


def printResult():
    letters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']   # predefined letters in data.py
    text = []                                       # annotations for the graph (which letter belongs to which node)

    plt.figure(1)
    for x in range(wd):
        text.append([])
        for y in range(ln):
            text[x].append([])
            plt.gca().add_patch(plt.Rectangle((x*10, y*10), height=10, width=10, fc='white', ec='black'))

    # iteration through fonts
    for f in range(len(inp)):
        # iteration through letters
        for l in range(len(inp[f])):
            # find the closest node
            minX = minY = 0
            minD = 1000000
            for x in range(len(w)):
                for y in range(len(w[x])):
                    dist = 0
                    for i in range(len(w[x][y])):
                        dist += abs(w[x][y][i] - inp[f][l][i])

                    if dist < minD:
                        minD = dist
                        minX = x
                        minY = y

            if not text[minX][minY]:
                text[minX][minY] = plt.text(minX * 10 + 5, minY * 10 + 5, letters[l] + str(f + 1),
                                            color='black', fontsize=7, ha='center', va='center')
            else:
                text[minX][minY].set_text(text[minX][minY].get_text() + ", " + letters[l] + str(f + 1))

    plt.xticks(range(5, wd*10 - 4, 10), range(1, wd + 1))
    plt.yticks(range(5, ln*10 - 4, 10), range(1, ln + 1))
    plt.axis('scaled')
    plt.show()


def main():
    init()
    train()
    printResult()


if __name__ == "__main__":
    main()
