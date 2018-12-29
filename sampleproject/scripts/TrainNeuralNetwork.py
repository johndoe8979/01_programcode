import numpy as np
import matplotlib.pylab as plt
from common.TwoLayerNet import TwoLayerNetwork
from mnistlist.dataset.mnist import load_mnist


network = TwoLayerNetwork(input_size=784, hidden_size=50, output_size=10, weight_init=0.01)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Set Variable and array
training_loss_list = []
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
loop_count = 10000

# Set array for loss rate calculation
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

print("Start Script")

# Main method
for i in range(loop_count):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.get_numerical_gradient(x_batch, t_batch)

    print("Loop Count : " + str(i))

    for key in ('first_weight', 'first_bias', 'second_weight', 'second_bias'):
        network.parameters[key] -= learning_rate * grad[key]

    loss = network.get_lossrate(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.get_accuracyrate(x_train, t_train)
        test_acc = network.get_accuracyrate(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# Show Transition of loss rate
markers = {'train': 'o', 'test': 's'}
x = np.array(len(train_acc_list))
plt.plot(x, train_acc_list, label='train accuracy')
plt.plot(x, test_acc_list, label='test_accuracy', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
