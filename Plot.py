from matplotlib import pyplot as plt
import numpy as np
import itertools


def generate_plot(his, mdl):
    plt.title('Accuracy and Loss (' + mdl + ')')
    plt.plot(his.history['acc'], color='r', label='Training Accuracy')
    # plt.plot(his.history['loss'], color='g', label='Training Loss')
    plt.legend(loc='best')
    plt.tight_layout()


def draw_history(his, mdl, file_path):
    generate_plot(his, mdl)
    plt.savefig(file_path)
