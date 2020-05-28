import datetime
from art.classifiers import KerasClassifier
from art.metrics import empirical_robustness, clever
from art.attacks import DeepFool, FastGradientMethod, ElasticNet
from keras.losses import categorical_crossentropy
import gc

from problems import *

from forBernoulli import *
from forExpectation import *

def attackCheck(classifier):
  results = {}
#   adv_crafter = DeepFool(classifier)
#   print('generating Deep Fool data ...')
#   DeepFool_x_test_adv = adv_crafter.generate(x_test)
  adv_crafter = FastGradientMethod(classifier, eps=0.1)
  print('generating FGM data ...')
  FastGradientMethod_x_test_adv = adv_crafter.generate(x=x_test)

#   return DeepFool_x_test_adv, FastGradientMethod_x_test_adv
  return x_test, FastGradientMethod_x_test_adv


def do_multitest_exp(problem, modelfunction, loadWeightsFrom):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    # Read dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, batch_size = loadProblem(problem)

    print('startTime: ' + datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_'))
    res = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, keras.activations.linear, t4e=1)
    model.load_weights(loadWeightsFrom)
    classifier = KerasClassifier(model=model) # Attack classifier
    DF_x_test, FGM_x_test = attackCheck(classifier)
    # model = load_model(loadWeightsFrom, custom_objects={'custom_activation_t1':custom_activation_t1, 'EConv2D':EConv2D})
    print('linear ...')
    res['linear'] = {'FGSM_Attack_res': model.evaluate_generator(rep_input_flow(FGM_x_test, y_test, batch_size=batch_size, t=1),
                                                      steps=len(y_train) // batch_size + 1, verbose=2),
                     'DeepFool_Attack_res': model.evaluate_generator(rep_input_flow(DF_x_test, y_test, batch_size=batch_size, t=1),
                                                      steps=len(y_train) // batch_size + 1, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=1)
    model.load_weights(loadWeightsFrom)
    print('t1 ...')
    res['t1'] = {'FGSM_Attack_res': model.evaluate_generator(rep_input_flow(FGM_x_test, y_test, batch_size=batch_size, t=1),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'DeepFool_Attack_res': model.evaluate_generator(rep_input_flow(DF_x_test, y_test, batch_size=batch_size, t=1),
                                                 steps=len(y_test) // batch_size + 1, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=2)
    model.load_weights(loadWeightsFrom)
    print('t2 ...')
    res['t2'] = {'FGSM_Attack_res': model.evaluate_generator(rep_input_flow(FGM_x_test, y_test, batch_size=batch_size, t=2),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'DeepFool_Attack_res': model.evaluate_generator(rep_input_flow(DF_x_test, y_test, batch_size=batch_size, t=2),
                                                 steps=len(y_test) // batch_size + 1, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=3)
    model.load_weights(loadWeightsFrom)
    print('t3 ...')
    res['t3'] = {'FGSM_Attack_res': model.evaluate_generator(rep_input_flow(FGM_x_test, y_test, batch_size=batch_size, t=3),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'DeepFool_Attack_res': model.evaluate_generator(rep_input_flow(DF_x_test, y_test, batch_size=batch_size, t=3),
                                                 steps=len(y_test) // batch_size + 1, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1, t4e=5)
    model.load_weights(loadWeightsFrom)
    print('t5 ...')
    res['t5'] = {'FGSM_Attack_res': model.evaluate_generator(rep_input_flow(FGM_x_test, y_test, batch_size=batch_size, t=5),
                                                 steps=len(y_test) // batch_size + 1, verbose=2),
                 'DeepFool_Attack_res': model.evaluate_generator(rep_input_flow(DF_x_test, y_test, batch_size=batch_size, t=5),
                                                 steps=len(y_test) // batch_size + 1, verbose=2)}

    batch_size = None
    num_classes = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    input_shape = None

    gc.collect()

    return res


def do_multitest_no_augment(problem, modelfunction, loadWeightsFrom):
    global batch_size, num_classes, x_train, y_train, x_test, y_test, input_shape, now

    # Read dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, batch_size = loadProblem(problem)

    print('startTime: ' + datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_'))
    res = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}

    model, funame = modelfunction(input_shape, num_classes, keras.activations.linear)
    model.load_weights(loadWeightsFrom)
    classifier = KerasClassifier(model=model) # Atack classifire
    DF_x_test, FGM_x_test = attackCheck(classifier)
    print('linear ...')
    res['linear'] = {'FGSM_Attack_res': model.evaluate(FGM_x_test, y_test, verbose=2),
                     'DeepFool_Attack_res': model.evaluate(DF_x_test, y_test, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t1)
    model.load_weights(loadWeightsFrom)
    print('t1 ...')
    res['t1'] = {'FGSM_Attack_res': model.evaluate(FGM_x_test, y_test, verbose=2),
                 'DeepFool_Attack_res': model.evaluate(DF_x_test, y_test, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t2)
    model.load_weights(loadWeightsFrom)
    print('t2 ...')
    res['t2'] = {'FGSM_Attack_res': model.evaluate(FGM_x_test, y_test, verbose=2),
                 'DeepFool_Attack_res': model.evaluate(DF_x_test, y_test, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t3)
    model.load_weights(loadWeightsFrom)
    print('t3 ...')
    res['t3'] = {'FGSM_Attack_res': model.evaluate(FGM_x_test, y_test, verbose=2),
                 'DeepFool_Attack_res': model.evaluate(DF_x_test, y_test, verbose=2)}

    model, funame = modelfunction(input_shape, num_classes, custom_activation_t5)
    model.load_weights(loadWeightsFrom)
    print('t5 ...')
    res['t5'] = {'FGSM_Attack_res': model.evaluate(FGM_x_test, y_test, verbose=2),
                 'DeepFool_Attack_res': model.evaluate(DF_x_test, y_test, verbose=2)}

    batch_size = None
    num_classes = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    input_shape = None

    gc.collect()

    return res



def TestCifar10():
    db = 'cifar10'
    #  -------------------------------------------------- Expectation Binomial -----------------------------------------
    exp_multitest = {}
    print('exp: linear')
    exp_multitest['linear'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # linear
                                               loadWeightsFrom='results/StartAt_190428-164814_cifar10_t1_net1_expectation_binomial_residual_network/e_weights/00000210.h5')

    print('exp: t1')
    exp_multitest['t1'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t1
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t1_net1_expectation_binomial_residual_network/e_weights/00000190.h5')
    print('exp: t2')
    exp_multitest['t2'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t2
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t2_net1_expectation_binomial_residual_network/e_weights/00000210.h5')
    print('exp: t3')
    exp_multitest['t3'] = do_multitest_exp(db, net1_expectation_binomial_residual_network,  # t3
                                           loadWeightsFrom='results/StartAt_190506-100246_cifar10_t3_net1_expectation_binomial_residual_network/e_weights/00000160.h5')

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + '_exp_multitest_' + serie + '.pkl', 'wb') as f:
        pickle.dump(exp_multitest, f, pickle.HIGHEST_PROTOCOL)
    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {}
    print('no_augment: linear')
    noaug_multitest['linear'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # linear
                                                        loadWeightsFrom='results/StartAt_190428-164814_cifar10_linear_net2_bernoulli_residual_network_no_augments/e_weights/00000080.h5')
    print('no_augment: t1')
    noaug_multitest['t1'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t1
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t1_net2_bernoulli_residual_network_no_augments/e_weights/00000220.h5')
    print('no_augment: t2')
    noaug_multitest['t2'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t2
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t2_net2_bernoulli_residual_network_no_augments/e_weights/00000240.h5')
    print('no_augment: t3')
    noaug_multitest['t3'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t3
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t3_net2_bernoulli_residual_network_no_augments/e_weights/00000240.h5')
    print('no_augment: t5')
    noaug_multitest['t5'] = do_multitest_no_augment(db, net2_bernoulli_residual_network,  # t5
                                                    loadWeightsFrom='results/StartAt_190428-164814_cifar10_t5_net2_bernoulli_residual_network_no_augments/e_weights/00000110.h5')

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'noaug_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(noaug_multitest, f, pickle.HIGHEST_PROTOCOL)


def TestMNIST():
    db = 'mnist'
    #  -------------------------------------------------- Expectation Binomial -----------------------------------------
    exp_multitest = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}
    print('exp: linear')
    exp_multitest['linear'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # linear
                                               loadWeightsFrom='results/StartAt_190528-094208_mnist_linear_net1_expectation_binomial_residual_network/e_weights/00000070.h5'),

    print('exp: t1')
    exp_multitest['t1'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t1
                                           loadWeightsFrom='results/StartAt_190528-094208_mnist_t1_net1_expectation_binomial_residual_network/e_weights/00000120.h5'),
    print('exp: t2')
    exp_multitest['t2'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t2
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t2_net1_expectation_binomial_residual_network/e_weights/00000160.h5'),
    print('exp: t3')
    exp_multitest['t3'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t3
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t3_net1_expectation_binomial_residual_network/e_weights/00000200.h5'),
    print('exp: t4')
    exp_multitest['t4'] = do_multitest_exp(db, net1mnist_expectation_binomial_residual_network,  # t4
                                           loadWeightsFrom='results/StartAt_190525-002923_mnist_t4_net1_expectation_binomial_residual_network/e_weights/00000140.h5'),

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'exp_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(exp_multitest, f, pickle.HIGHEST_PROTOCOL)
    #  ----------------------------------------------------------- Binomial -------------------------------------------
    noaug_multitest = {'startTime': datetime.datetime.now().strftime('StartAt_%y%m%d-%H%M%S_')}
    print('no_augment: linear')
    noaug_multitest['linear'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # linear
                                                        loadWeightsFrom='results/StartAt_190529-114807_mnist_linear_net2_bernoulli_residual_network_no_augments/e_weights/00000230.h5'),
    print('no_augment: t1')
    noaug_multitest['t1'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t1
                                                    loadWeightsFrom='results/StartAt_190528-094208_mnist_t1_net2_bernoulli_residual_network_no_augments/e_weights/00000120.h5'),
    print('no_augment: t2')
    noaug_multitest['t2'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t2
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t2_net2_bernoulli_residual_network_no_augments/e_weights/00000230.h5'),
    print('no_augment: t3')
    noaug_multitest['t3'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t3
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t3_net2_bernoulli_residual_network_no_augments/e_weights/00000040.h5'),
    print('no_augment: t5')
    noaug_multitest['t5'] = do_multitest_no_augment(db, net2mnist_bernoulli_residual_network,  # t5
                                                    loadWeightsFrom='results/StartAt_190529-114807_mnist_t5_net2_bernoulli_residual_network_no_augments/e_weights/00000210.h5'),

    serie = datetime.datetime.now().strftime('_%y%m%d-%H%M%S')
    with open(db + 'noaug_multitest' + serie + '.pkl', 'wb') as f:
        pickle.dump(noaug_multitest, f, pickle.HIGHEST_PROTOCOL)


TestCifar10()
TestMNIST()
