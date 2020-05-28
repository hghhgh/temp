import json
import matplotlib.pyplot as plt
import numpy as np


def loadLoss(filaddr):
    with open(filaddr, 'r') as f:
        data = f.readlines()
        return json.loads(data[0].replace('\'', '\"'))


def draw4cifar10():
    supTitle = 'CIFAR10'
    eleman = 'val_acc'

    exp_multitest = {}
    exp_multitest['linear'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_t1_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t1'] = loadLoss(
        'results/StartAt_190506-100246_cifar10_t1_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t2'] = loadLoss(
        'results/StartAt_190506-100246_cifar10_t2_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t3'] = loadLoss(
        'results/StartAt_190506-100246_cifar10_t3_net1_expectation_binomial_residual_network/e_loss_history.txt')

    plt.grid(True)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=25)
    plt.ylabel('Test Accuracy(%)', fontsize=25)
    plt.plot(np.asarray(exp_multitest['linear'][eleman]) * 100, linestyle="-", color="r", label='linear')
    plt.plot(np.asarray(exp_multitest['t1'][eleman]) * 100, linestyle="--", color="g", label='t=1')
    plt.plot(np.asarray(exp_multitest['t2'][eleman]) * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(np.asarray(exp_multitest['t3'][eleman]) * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.legend(loc="lower right")
    plt.title('Training Type A(' + supTitle + ')')
    plt.savefig('pics/' + supTitle + '_Type_A_Training.png', bbox_inches='tight', dpi=300)
    plt.show()

    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {}
    noaug_multitest['linear'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_linear_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t1'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_t1_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t2'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_t2_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t3'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_t3_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t5'] = loadLoss(
        'results/StartAt_190428-164814_cifar10_t5_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')

    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=25)
    plt.ylabel('Test Accuracy(%)', fontsize=25)
    plt.plot(np.asarray(noaug_multitest['linear'][eleman]) * 100, linestyle="-", color="r", label='linear')
    plt.plot(np.asarray(noaug_multitest['t1'][eleman]) * 100, linestyle="--", color="g", label='t=1')
    plt.plot(np.asarray(noaug_multitest['t2'][eleman]) * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(np.asarray(noaug_multitest['t3'][eleman]) * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.legend(loc="lower right")
    plt.title('Training Type B(' + supTitle + ')')
    plt.savefig('pics/' + supTitle + '_Type_B_Training.png', bbox_inches='tight', dpi=300)
    plt.show()


def draw4mnist():
    supTitle = 'MNIST'
    eleman = 'val_acc'

    #  -------------------------------------------------- Expectation Binomial -----------------------------------------
    exp_multitest = {}
    exp_multitest['linear'] = loadLoss(
        'results/StartAt_190528-094208_mnist_linear_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t1'] = loadLoss(
        'results/StartAt_190528-094208_mnist_t1_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t2'] = loadLoss(
        'results/StartAt_190525-002923_mnist_t2_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t3'] = loadLoss(
        'results/StartAt_190525-002923_mnist_t3_net1_expectation_binomial_residual_network/e_loss_history.txt')
    exp_multitest['t4'] = loadLoss(
        'results/StartAt_190525-002923_mnist_t4_net1_expectation_binomial_residual_network/e_loss_history.txt')

    plt.grid(True)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=25)
    plt.ylabel('Test Accuracy(%)', fontsize=25)
    plt.plot(np.asarray(exp_multitest['linear'][eleman]) * 100, linestyle="-", color="r", label='linear')
    plt.plot(np.asarray(exp_multitest['t1'][eleman]) * 100, linestyle="--", color="g", label='t=1')
    plt.plot(np.asarray(exp_multitest['t2'][eleman]) * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(np.asarray(exp_multitest['t3'][eleman]) * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.legend(loc="lower right")
    plt.title('Training Type A(' + supTitle + ')')
    plt.savefig('pics/' + supTitle + '_Type_A_Training.png', bbox_inches='tight', dpi=300)
    plt.show()

    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {}
    noaug_multitest['linear'] = loadLoss(
        'results/StartAt_190529-114807_mnist_linear_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t1'] = loadLoss(
        'results/StartAt_190528-094208_mnist_t1_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t2'] = loadLoss(
        'results/StartAt_190529-114807_mnist_t2_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t3'] = loadLoss(
        'results/StartAt_190529-114807_mnist_t3_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')
    noaug_multitest['t5'] = loadLoss(
        'results/StartAt_190529-114807_mnist_t5_net2_bernoulli_residual_network_no_augments/e_loss_history.txt')

    plt.figure(figsize=(12, 6), dpi=100)
    plt.xlabel('Number of epochs', fontsize=25)
    plt.ylabel('Test Accuracy(%)', fontsize=25)
    plt.plot(np.asarray(noaug_multitest['linear'][eleman]) * 100, linestyle="-", color="r", label='linear')
    plt.plot(np.asarray(noaug_multitest['t1'][eleman]) * 100, linestyle="--", color="g", label='t=1')
    plt.plot(np.asarray(noaug_multitest['t2'][eleman]) * 100, linestyle="-.", color="b", label='t=2')  # acc
    plt.plot(np.asarray(noaug_multitest['t3'][eleman]) * 100, linestyle=":", color="m", label='t=3')  # acc
    plt.legend(loc="lower right")
    plt.title('Training Type B(' + supTitle + ')')
    plt.savefig('pics/' + supTitle + '_Type_B_Training.png', bbox_inches='tight', dpi=300)
    plt.show()


# draw4cifar10()
draw4mnist()