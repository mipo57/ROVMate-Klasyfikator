
import math
import cv2
import glob
import numpy as np
import os.path
import os
import tables as tb

class Dataset:
    valid = 0.2
    classes = [("A", 0), ("B", 1), ("C", 2), ("D", 3), ("E", 4), ("F", 5)]
    input_shape = (416, 416, 3)
    label_shape = (13, 13, 11)
    train_label_shape = (5,)
    filename = "dataset.hdf5"

    def prepare(self):
        if os.path.exists(self.filename):
            print("Deleting old dataset...")
            os.remove(self.filename)
            print("Done")

        print("Creating dataset file...")
        output_file = tb.open_file(self.filename, mode="w")

        type = tb.Float32Atom()
        train_set = output_file.create_earray(output_file.root, "train_set", type, (0, *self.input_shape))
        valid_set = output_file.create_earray(output_file.root, "valid_set", type, (0, *self.input_shape))

        train_labels = output_file.create_earray(output_file.root, "train_labels", type, (0, *self.train_label_shape))
        valid_labels = output_file.create_earray(output_file.root, "valid_labels", type, (0, *self.label_shape))
        print("Done")

        num_data = len(glob.glob("ren_*/*") )

        print("Exporting...")
        i = 0
        for cls in self.classes:
            ren_folder = "ren_{}".format(cls[0])
            seg_folder = "seg_{}".format(cls[0])

            for file in glob.glob("{}/*".format(seg_folder)):
                print("{} / {}".format(i, num_data))
                image_ren = cv2.imread("{}/{}".format(ren_folder, file.split("/")[-1])).astype(np.float32)
                image_seg = cv2.imread(file)

                img_width = np.shape(image_seg)[1]
                img_height = np.shape(image_seg)[0]
                tile_width = img_width / 13
                tile_height = img_height / 13

                min_x = math.inf
                max_x = -math.inf
                min_y = math.inf
                max_y = -math.inf

                for y in range(int(img_height)):
                    for x in range(int(img_width)):
                        if np.any(image_seg[y, x, :] != [0, 0, 0]):
                            if x > max_x:
                                max_x = x
                            if x < min_x:
                                min_x = x
                            if y > max_y:
                                max_y = y
                            if y < min_y:
                                min_y = y

                label_train = [[min_x, min_y, max_x, max_y, cls[1]]]
                label = np.zeros((13, 13, 11))

                bb_width = (max_x - min_x) / img_width
                bb_height = (max_y - min_y) / img_height

                x = (min_x + max_x) / 2
                y = (min_y + max_y) / 2

                tile_x = int(x // tile_width)
                tile_y = int(y // tile_height)

                label[tile_y, tile_x, 0] = (x - tile_x * tile_width) / tile_width
                label[tile_y, tile_x, 1] = (y - tile_y * tile_height) / tile_height
                label[tile_y, tile_x, 2] = bb_width
                label[tile_y, tile_x, 3] = bb_height
                label[tile_y, tile_x, 4] = 1
                label[tile_y, tile_x, 5 + cls[1]] = 1

                rnd = np.random.rand()

                image_ren = np.reshape(image_ren, (1, *image_ren.shape))
                label = np.reshape(label, (1, *label.shape))

                if rnd < self.valid:
                    valid_set.append(image_ren)
                    valid_labels.append(label)
                else:
                    train_set.append(image_ren)
                    train_labels.append(label_train)

                i = i + 1

        output_file.close()
        print("Done")


dataset = Dataset()
dataset.prepare()
