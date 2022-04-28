import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import config.log as log

logger = log.setup_custom_logger(__name__)


def gpu_config(model):
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_gpu:
        if gpu_count > 1:
            print(f'use {gpu_count} gpu who named:')
            for i in range(gpu_count):
                print(torch.cuda.get_device_name(i))
            model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
        else:
            print(f'use 1 gpu who named: {torch.cuda.get_device_name(0)}')
    else:
        print('no gpu available !')
    model.to(device)
    return model, device


def print_log(_output, _engine, _fp):
    output_items = " - ".join([f"{m}:{v:.4f}" for m, v in _output.items()])
    msg = f"{_engine.state.epoch} | {_engine.state.iteration}: {output_items}"

    with open(_fp, "a") as h:
        h.write(msg)
        h.write("\n")
    return


def log_model_weights(engine, model=None, fp=None):
    """Helper method to tensorboard_logs norms of model weights: print and dump into a file"""
    assert model and fp
    output = {"total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        name = name.replace(".", "/")
        n = torch.norm(p)
        if max_counter > 0:
            output[name] = n
        output["total"] += n
        max_counter -= 1
    print_log(output, engine, fp)
    return


def log_model_grads(engine, model=None, fp=None):
    """Helper method to tensorboard_logs norms of model gradients: print and dump into a file"""
    assert model and fp
    output = {"grads/total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        name = name.replace(".", "/")
        n = torch.norm(p.grad)
        if max_counter > 0:
            output[f"grads/{name}"] = n
        output["grads/total"] += n
        max_counter -= 1
    print_log(output, engine, fp)
    return


def log_data_stats(engine, fp=None, **kwargs):
    """Helper method to tensorboard_logs mean/std of input batch of images and median of batch of targets."""
    assert fp
    x, y = engine.state.batch
    output = {
        "batch xmean": x.mean().item(),
        "batch xstd": x.std().item(),
        "batch ymedian": y.median().item(),
    }
    print_log(output, engine, fp)
    return


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def accuracy(dataloader, model, n_img):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f'Accuracy of the network on the {n_img} test images: {100 * correct // total} %')
    return


def accuracy_per_classes(dataloader, model, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        _accuracy = 100 * float(correct_count) / total_pred[classname]
        logger.info(f'Accuracy for class: {classname:5s} is {_accuracy:.1f} %')
    return
