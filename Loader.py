import os
import sys
import patoolib


def Load_TestList_and_labels(test_path, data_dir_path, labels):
    input_data_dir_path = data_dir_path + test_path

    result = dict()

    dir_count = 0
    for label in labels:
        file_path = input_data_dir_path + os.path.sep + label
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = label
    return result


def main():
    print('ok')


if __name__ == '__main__':
    main()
