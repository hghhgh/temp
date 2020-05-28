import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import json


def loadArNo0(fname, thr=0.0, startidx=0):
    file = open(fname, 'r')
    fstr = file.read().replace("\'", "\"")
    arr = json.loads(fstr)

    return arr


def drawChart4(sidx, linear_test, supTitle, t1_test, t2_test, t3_test):
    ws = np.asarray(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
         250])
    mlti = ws[np.argmax(linear_test[ws - 1])]
    mlt = round(linear_test[mlti - 1] * 100, 2)
    mt1i = ws[np.argmax(t1_test[ws - 1])]
    mt1 = round(t1_test[mlti - 1] * 100, 2)
    mt2i = ws[np.argmax(t2_test[ws - 1])]
    mt2 = round(t2_test[mlti - 1] * 100, 2)
    mt3i = ws[np.argmax(t3_test[ws - 1])]
    mt3 = round(t3_test[mlti - 1] * 100, 2)
    print '------------- expectation binomial----------------'
    print 'linear:' + str(mlt) + ' t1:' + str(mt1) + ' t2:' + str(mt2) + ' t3:' + str(mt3)
    print 'lineari:' + str(mlti) + ' t1i:' + str(mt1i) + ' t2i:' + str(mt2i) + ' t3i:' + str(mt3i)
    fsize = 25
    mlen = len(linear_test)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=fsize)
    plt.ylabel('Accuracy(%)', fontsize=fsize)
    # plt.title('Test Accuracy of ' + supTitle)
    plt.grid(True)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(np.arange(0, mlen, step=10), np.arange(sidx, mlen + sidx, step=10))
    plt.plot(linear_test * 100, linestyle="-", color="r", label='linear')
    plt.plot(t1_test * 100, linestyle="--", color="g", label='t=1')
    plt.plot(t2_test * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(t3_test * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.legend(loc="lower right")
    plt.savefig(supTitle + '_Test_Accuracy.png', bbox_inches='tight', dpi=300)
    plt.show()


def drawChart5(sidx, linear_test, supTitle, t1_test, t2_test, t3_test, t5_test):
    ws = np.asarray(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
         250])
    mlti = ws[np.argmax(linear_test[ws - 1])]
    mlt = round(linear_test[mlti - 1] * 100, 2)
    mt1i = ws[np.argmax(t1_test[ws - 1])]
    mt1 = round(t1_test[mlti - 1] * 100, 2)
    mt2i = ws[np.argmax(t2_test[ws - 1])]
    mt2 = round(t2_test[mlti - 1] * 100, 2)
    mt3i = ws[np.argmax(t3_test[ws - 1])]
    mt3 = round(t3_test[mlti - 1] * 100, 2)
    mt5i = ws[np.argmax(t5_test[ws - 1])]
    mt5 = round(t5_test[mlti - 1] * 100, 2)
    print '------------- binomial----------------'
    print 'linear:' + str(mlt) + ' t1:' + str(mt1) + ' t2:' + str(mt2) + ' t3:' + str(mt3) + ' t5:' + str(mt5)
    print 'lineari:' + str(mlti) + ' t1i:' + str(mt1i) + ' t2i:' + str(mt2i) + ' t3i:' + str(mt3i) + ' t5i:' + str(mt5i)
    fsize = 25
    mlen = len(linear_test)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=fsize)
    plt.ylabel('Accuracy(%)', fontsize=fsize)
    # plt.title('Test Accuracy of ' + supTitle)
    plt.grid(True)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(np.arange(0, mlen, step=10), np.arange(sidx, mlen + sidx, step=10))
    plt.plot(linear_test * 100, linestyle="-", color="r", label='linear')
    plt.plot(t1_test * 100, linestyle="--", color="g", label='t=1')
    plt.plot(t2_test * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(t3_test * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.plot(t5_test * 100, linestyle="-", color="b", label='t=5')  # acc
    plt.legend(loc="lower right")
    plt.savefig(supTitle + '_Test_Accuracy.png', bbox_inches='tight', dpi=300)
    plt.show()


def drawChart5m(sidx, linear_test, supTitle, t1_test, t2_test, t3_test, t4_test):
    ws = np.asarray(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
         250])
    mlti = ws[np.argmax(linear_test[ws - 1])]
    mlt = round(linear_test[mlti - 1] * 100, 2)
    mt1i = ws[np.argmax(t1_test[ws - 1])]
    mt1 = round(t1_test[mlti - 1] * 100, 2)
    mt2i = ws[np.argmax(t2_test[ws - 1])]
    mt2 = round(t2_test[mlti - 1] * 100, 2)
    mt3i = ws[np.argmax(t3_test[ws - 1])]
    mt3 = round(t3_test[mlti - 1] * 100, 2)
    mt4i = ws[np.argmax(t4_test[ws - 1])]
    mt4 = round(t4_test[mlti - 1] * 100, 2)
    print '------------- expectation----------------'
    print 'linear:' + str(mlt) + ' t1:' + str(mt1) + ' t2:' + str(mt2) + ' t3:' + str(mt3) + ' t4:' + str(mt4)
    print 'lineari:' + str(mlti) + ' t1i:' + str(mt1i) + ' t2i:' + str(mt2i) + ' t3i:' + str(mt3i) + ' t4i:' + str(mt4i)
    fsize = 25
    mlen = len(linear_test)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=fsize)
    plt.ylabel('Accuracy(%)', fontsize=fsize)
    # plt.title('Test Accuracy of ' + supTitle)
    plt.grid(True)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(np.arange(0, mlen, step=10), np.arange(sidx, mlen + sidx, step=10))
    plt.plot(linear_test * 100, linestyle="-", color="r", label='linear')
    plt.plot(t1_test * 100, linestyle="--", color="g", label='t=1')
    plt.plot(t2_test * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(t3_test * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.plot(t4_test * 100, linestyle="-", color="b", label='t=4')  # acc
    plt.legend(loc="lower right")
    plt.savefig(supTitle + '_Test_Accuracy.png', bbox_inches='tight', dpi=300)
    plt.show()


def drawCIFAR10_Expectation():
    thr = .0
    sidx = 0
    repSTRT = 5
    linear_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_t1_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t1_test = loadArNo0(
        'results/StartAt_190506-100246_cifar10_t1_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t2_test = loadArNo0(
        'results/StartAt_190506-100246_cifar10_t2_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t3_test = loadArNo0(
        'results/StartAt_190506-100246_cifar10_t3_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)

    supTitle = 'CIFAR10 Expectation'

    drawChart4(sidx, np.asarray(linear_test['val_acc']), supTitle, np.asarray(t1_test['val_acc']),
               np.asarray(t2_test['val_acc']), np.asarray(t3_test['val_acc']))


def drawCIFAR10_Binomial():
    thr = .0
    sidx = 0
    repSTRT = 5
    linear_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_linear_net2_bernoulli_residual_network_no_augments/e_loss_history.txt',
        thr,
        sidx)
    t1_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_t1_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t2_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_t2_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t3_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_t3_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t5_test = loadArNo0(
        'results/StartAt_190428-164814_cifar10_t5_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)

    supTitle = 'CIFAR10 Binomial'

    drawChart5(sidx, np.asarray(linear_test['val_acc']), supTitle, np.asarray(t1_test['val_acc']),
               np.asarray(t2_test['val_acc']), np.asarray(t3_test['val_acc']), np.asarray(t5_test['val_acc']))


def drawMNIST_Expectation():
    thr = .0
    sidx = 0
    repSTRT = 5
    linear_test = loadArNo0(
        'results/StartAt_190528-094208_mnist_linear_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t1_test = loadArNo0(
        'results/StartAt_190528-094208_mnist_t1_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t2_test = loadArNo0(
        'results/StartAt_190525-002923_mnist_t2_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t3_test = loadArNo0(
        'results/StartAt_190525-002923_mnist_t3_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)
    t4_test = loadArNo0(
        'results/StartAt_190525-002923_mnist_t4_net1_expectation_binomial_residual_network/e_loss_history.txt', thr,
        sidx)

    supTitle = 'MNIST Expectation'

    drawChart5m(sidx, np.asarray(linear_test['val_acc']), supTitle, np.asarray(t1_test['val_acc']),
               np.asarray(t2_test['val_acc']), np.asarray(t3_test['val_acc']), np.asarray(t4_test['val_acc']))


def drawMNIST_Binomial():
    thr = .0
    sidx = 0
    repSTRT = 5
    linear_test = loadArNo0(
        'results/StartAt_190529-114807_mnist_linear_net2_bernoulli_residual_network_no_augments/e_loss_history.txt',
        thr,
        sidx)
    t1_test = loadArNo0(
        'results/StartAt_190528-094208_mnist_t1_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t2_test = loadArNo0(
        'results/StartAt_190529-114807_mnist_t2_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t3_test = loadArNo0(
        'results/StartAt_190529-114807_mnist_t3_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)
    t5_test = loadArNo0(
        'results/StartAt_190529-114807_mnist_t5_net2_bernoulli_residual_network_no_augments/e_loss_history.txt', thr,
        sidx)

    supTitle = 'MNIST Binomial'

    drawChart5(sidx, np.asarray(linear_test['val_acc']), supTitle, np.asarray(t1_test['val_acc']),
               np.asarray(t2_test['val_acc']), np.asarray(t3_test['val_acc']), np.asarray(t5_test['val_acc']))


drawCIFAR10_Expectation()
drawCIFAR10_Binomial()

drawMNIST_Expectation()
drawMNIST_Binomial()
