import matplotlib.pyplot as plt


def between_training_and_testing_loss(train_loss , test_loss):
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def between_training_and_testing_acc(train_loss , test_loss , train_accuracy , test_accuracy):
    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()