import argparse
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from models import train_model_lenet
from models import train_model_nvidia
from models import train_model_comma

# Generator to load process batch size of images
def generator(data_path, samples, batch_size):
    sklearn.utils.shuffle(samples)
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            images = []
            measurements = []
            # Process batch_size of images
            for line in samples[offset:offset+batch_size]:
                for i in range(3):
                    # Append image
                    img_name = line[i].split("/")[-1]
                    img_path = data_path + "IMG/" + img_name
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    # Append steering measurement
                    steering = float(line[3])
                    if i==1:
                        steering = steering + 0.2
                    elif i==2:
                        steering = steering - 0.2
                    measurements.append(steering)
                    # Append flipped image
                    img_flipped = cv2.flip(img, 1)
                    images.append(img_flipped)
                    measurements.append(-steering)

            x_data = np.array(images)
            y_data = np.array(measurements)
            yield(sklearn.utils.shuffle(x_data, y_data))

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser(description = 'Train Model')
    parser.add_argument(
        'data',
        type = str,
        help = 'Path to folder containing training images and driving_log'
    )
    parser.add_argument(
        'model',
        type = str,
        help = 'Model Name: lenet, nvidia, comma'
    )
    args = parser.parse_args()

    # Read csv file
    file_path = args.data + "driving_log.csv"
    lines = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    # Delete column headings
    del(lines[0])

    # Split dataset into training and validation set
    train_samples, val_samples = train_test_split(lines, test_size=0.3)
    batch_size = 16
    train_generator = generator(args.data, train_samples, batch_size)
    val_generator = generator(args.data, val_samples, batch_size)

    # Call function to train specific model
    if args.model == 'lenet':
        print("Network Architecture = LeNet-5")
        train_model_lenet(train_generator, val_generator, \
                            len(train_samples), len(val_samples))
    elif args.model == 'nvidia':
        print("Network Architecture = Nvidia")
        train_model_nvidia(train_generator, val_generator, \
                            len(train_samples), len(val_samples))
    elif args.model == 'comma':
        print("Network Architecture = Comma.ai")
        train_model_comma(train_generator, val_generator, \
                            len(train_samples), len(val_samples))
