import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import imgaug as ia
import matplotlib.pyplot as plt
import tables
import math
import tqdm

PATH = "./skrzydlo/"
BATCH_SIZE = 5
IMG_SIZE = 416

COORD = 0#5
NOOBJ = 0.1

dane = pd.read_csv("adnotacje-skrzydlo2.csv", header=None)

pliki = dane[0].as_matrix()
adnotacje = np.reshape(dane.drop([0], axis=1).as_matrix(), (len(pliki), 4))

LUT = dict([(0, "Czerwony trojkat"),
            (1, "Zolty trojkat"),
            (2, "Niebieski trojkat"),
            (3, "Czerwony trapez"),
            (4, "Zolty trapez"),
            (5, "Niebieski trapez"),
            ])

ia.seed(1)
seq = ia.augmenters.Sequential([
    ia.augmenters.Fliplr(0.5), 
    ia.augmenters.Crop(percent=(0, 0.1)), 
    ia.augmenters.Sometimes(0.5,
        ia.augmenters.GaussianBlur(sigma=(0, 0.5))
    ),
    ia.augmenters.ContrastNormalization((0.75, 1.5)),
    ia.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    ia.augmenters.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-25, 25),
        shear=(-4, 4)
    )
], random_order=True)


class Generator:
    def generate_data(self):
        pass


class TrainGenerator (Generator):
    source = None
    filename = "dataset.hdf5"

    def __init__(self):
        super()

        self.source = tables.open_file(self.filename, "r")
        self.train_set = self.source.root.train_set
        self.valid_set = self.source.root.valid_set

        self.train_labels = self.source.root.train_labels
        self.valid_labels = self.source.root.valid_labels

        self.num_train = self.train_set.shape[0]
        self.num_valid = self.valid_set.shape[0]

    def __del__(self):
        self.source.close()

    def generate_data(self):
        i = 0

        while True:
            x_batch = []
            y_batch = []

            indexes = np.arange(self.num_train)
            np.random.shuffle(indexes)

            for b in range(BATCH_SIZE):
                if i >= self.num_train:
                    np.random.shuffle(indexes)
                    i = 0

                img = self.train_set[indexes[i]]
                label = self.train_labels[indexes[i]]

                coords = label[:4]
                clas = label[4]

                bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(coords[0], coords[1], coords[2], coords[3])], img.shape)

                seq_det = seq.to_deterministic()
                img = seq_det.augment_image(image=img)
                bbs = seq_det.augment_bounding_boxes([bbs])[0]
                img = img.astype(np.float) / 255

                tile_size = IMG_SIZE // 13

                grid_x = int(bbs.bounding_boxes[0].center_x // (tile_size+1))
                grid_y = int(bbs.bounding_boxes[0].center_y // (tile_size+1))

                grid_x = 0 if grid_x < 0 else grid_x
                grid_y = 0 if grid_y < 0 else grid_y
                grid_x = 12 if grid_x > 12 else grid_x
                grid_y = 12 if grid_y > 12 else grid_y

                label = np.zeros((13, 13, 11), dtype=np.float32)
                label[grid_y, grid_x, 0] = bbs.bounding_boxes[0].center_x / tile_size - grid_x
                label[grid_y, grid_x, 1] = bbs.bounding_boxes[0].center_y / tile_size - grid_y
                label[grid_y, grid_x, 2] = bbs.bounding_boxes[0].width / IMG_SIZE
                label[grid_y, grid_x, 3] = bbs.bounding_boxes[0].height / IMG_SIZE
                label[grid_y, grid_x, 4] = 1
                label[:, :, 5 + int(clas)] = 1

                x_batch.append(img)
                y_batch.append(label)
                i += 1

            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)

            yield x_batch, y_batch


class ValidGenerator(Generator):
    source = None
    filename = "dataset.hdf5"

    def __init__(self):
        super()

        self.source = tables.open_file(self.filename, "r")
        self.train_set = self.source.root.train_set
        self.valid_set = self.source.root.valid_set

        self.train_labels = self.source.root.train_labels
        self.valid_labels = self.source.root.valid_labels

        self.num_train = self.train_set.shape[0]
        self.num_valid = self.valid_set.shape[0]

    def __del__(self):
        self.source.close()

    def generate_data(self):
        i = 0

        while True:
            x_batch = []
            y_batch = []

            indexes = np.arange(self.num_valid)
            np.random.shuffle(indexes)

            for b in range(BATCH_SIZE):
                if i >= self.num_valid:
                    np.random.shuffle(indexes)
                    i = 0

                img = self.valid_set[indexes[i]] / 255
                label = self.valid_labels[indexes[i]]

                x_batch.append(img)
                y_batch.append(label)
                i += 1

            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)

            yield x_batch, y_batch

class Model:
    def __init__(self):
        self.session = tf.Session()
        self.xs = None
        self.ys = None
        self.outputs = None
        self.loss_op = None
        self.model_path = "./model"

        self._build_model()
        self._create_loss()

        self.saver = tf.train.Saver()

    def _build_model(self, scale_factor=1):
        # Input
        self.xs = tf.placeholder(dtype=tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))

        # Layer 1
        nn = tf.layers.conv2d(self.xs, 16, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (2, 2), "same")

        # Layer 2
        nn = tf.layers.conv2d(nn, 32, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (2, 2), "same")

        # Layer 3
        nn = tf.layers.conv2d(nn, 64, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (2, 2), "same")

        # Layer 4
        nn = tf.layers.conv2d(nn, 128, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (2, 2), "same")

        # Layer 5
        nn = tf.layers.conv2d(nn, 256, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (2, 2), "same")

        # Layer 6
        nn = tf.layers.conv2d(nn, 512, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.max_pooling2d(nn, (2, 2), (1, 1), "same")

        # Layer 7
        nn = tf.layers.conv2d(nn, 1024, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)
        nn = tf.layers.conv2d(nn, 5, (1, 1), padding="same")

        # Layer 8
        nn = tf.layers.conv2d(nn, 512, (3, 3), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.layers.batch_normalization(nn)
        nn = tf.nn.leaky_relu(nn, 0.1)

        # Layer 9
        nn = tf.layers.conv2d(nn, 11, (1, 1), padding="same", kernel_initializer=tf.initializers.variance_scaling(scale_factor))
        nn = tf.sigmoid(nn)

        first_part = nn[..., :5]
        second_part = tf.div(nn[..., 5:], tf.tile(tf.reshape(tf.reduce_sum(nn[..., 5:], 3) + 1e-13, (-1, 13, 13, 1)), (1, 1, 1, 6)))
        nn = tf.concat([first_part, second_part], 3)

        self.outputs = nn

    def _create_loss(self, warmup=False, imbalance_factor=0.005):
        self.ys = tf.placeholder(tf.float32, (None, 13, 13, 11))

        pred_x = self.outputs[..., 0]
        pred_y = self.outputs[..., 1]
        pred_w = self.outputs[..., 2]
        pred_h = self.outputs[..., 3]
        pred_o = self.outputs[..., 4]
        pred_c = self.outputs[..., 5:]

        true_x = self.ys[..., 0]
        true_y = self.ys[..., 1]
        true_w = self.ys[..., 2]
        true_h = self.ys[..., 3]
        true_o = self.ys[..., 4]
        true_c = self.ys[..., 5:]

        size_multiplier = 5

        if warmup:
            size_multiplier = 0.1

        loss_pos = size_multiplier * tf.reduce_mean( true_o * (tf.square(pred_x - true_x) + tf.square(pred_y - true_y)))
        loss_size = size_multiplier * tf.reduce_mean( true_o * (tf.square(tf.sqrt(pred_w) - tf.sqrt(true_w)) + tf.square(tf.sqrt(pred_h) - tf.sqrt(true_h))))
        loss_conf = tf.reduce_mean( true_o * tf.square(pred_o - true_o))
        loss_conf2 = imbalance_factor * tf.reduce_mean( (1-true_o) * tf.square(pred_o - true_o))

        loss_class = tf.reduce_sum(true_o*tf.reduce_sum(tf.square(pred_c - true_c), reduction_indices=3))

        self.loss_op = loss_conf + loss_conf2 + loss_pos + loss_size + loss_class

    def save(self):
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        self.saver.save(self.session, "{}/model.chkpt".format(self.model_path))

    def restore(self):
        self.saver.restore(self.session, "{}/model.chkpt".format(self.model_path))

    def train(self, generator: Generator):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss_op)
        self.session.run(tf.global_variables_initializer())
        # self.restore()
        epoch_size = 30
        i = 0
        e = 0
        batch_losses = []
        best_avg_loss = 1e10
        pbar = tqdm.tqdm(total=epoch_size)
        for x_batch, y_batch in generator.generate_data():
            _, loss = self.session.run([train_op, self.loss_op], feed_dict={self.xs: x_batch, self.ys: y_batch})
            pbar.update(1)

            if np.isnan(loss):
                print("Stopping training: NaN loss")
                break

            batch_losses.append(float(loss))
            if i > epoch_size:
                i = 0
                avg_loss = np.mean(batch_losses)
                if avg_loss < best_avg_loss:
                    print("Saving...")
                    best_avg_loss = avg_loss
                    self.save()
                print("\n------------------")
                print("Epoch {}: {}".format(e, avg_loss))
                print("Best: {}".format(best_avg_loss))
                print("------------------\n")
                batch_losses.clear()
                pbar.close()
                pbar = tqdm.tqdm(total=epoch_size)
                e = e + 1
            i += 1

    def test(self, generator: Generator):
        self.session.run(tf.global_variables_initializer())
        self.restore()

        for x_batch, y_batch in generator.generate_data():
            pred = self.session.run(self.outputs, feed_dict={self.xs: x_batch})[0]

            best_score = (0, 0, 0)
            for x in range(13):
                for y in range(13):
                    score = pred[y, x, 4] #* np.max(pred[y, x, 5:])
                    if score > best_score[0]:
                        best_score = (score, x, y)

            tile_x = best_score[1]
            tile_y = best_score[2]
            total_x = (tile_x + pred[int(tile_y), int(tile_x), 0]) * 416 / 13
            total_y = (tile_y + pred[int(tile_y), int(tile_x), 1]) * 416 / 13
            width = pred[int(tile_y), int(tile_x), 2] * 416
            height = pred[int(tile_y), int(tile_x), 3] * 416

            tst_img = np.reshape(x_batch[0], (416, 416, 3))
            tst_img = cv2.rectangle(tst_img, (int(total_x - width / 2), int(total_y - height / 2)),
                                    (int(total_x + width / 2), int(total_y + height / 2)), (255, 0, 255))
            print(LUT[np.argmax(pred[int(tile_y), int(tile_x), 5:])])
            cv2.imshow('test', tst_img)
            cv2.waitKey(0)


model = Model()
#generatorTrain = TrainGenerator()
#model.train(generatorTrain)
generatorValid = ValidGenerator()
model.test(generatorValid)

#def train(sess, loss_op, xs, ys, generator, new_model = True, epoch_size = 50, num_epochs = None, max_fail_count = 100, name="model"):
#    best_loss = np.inf
#    failed_count = 0
#    confirmed_count = 0
#    train_speed = 1e-4
#    epoch_num = 1
#
#    train_op = tf.train.GradientDescentOptimizer(train_speed).minimize(loss_op)
#    saver = initialize(sess)
#
#    if num_epochs is None:
#        num_epochs = np.inf
#
#    if not new_model:
#        restore(saver, sess, "./{}".format(name))
#
#    print("Training model \"{}\" at speed {}".format(name, train_speed))
#
#    for x_batch, y_batch in generator():
#        batch_losses = []
#        for e in range(epoch_size):
#            _, loss = sess.run([train_op, loss_op], feed_dict={xs: x_batch, ys: y_batch})
#            batch_losses.append(loss)
#
#        loss = np.mean(batch_losses)
#        print("Epoch {}: {}".format(epoch_num, loss))
#        print("-- Std: {}".format(np.std(batch_losses)))
#        print("-- Min: {}".format(np.min(batch_losses)))
#        print("-- Max: {}".format(np.max(batch_losses)))
#
#        if loss < best_loss:
#            save(saver, sess, "./{}".format(name))
#            best_loss = loss
#            failed_count = 0
#            confirmed_count += 1
#            if confirmed_count > 15:
#                confirmed_count = 0
#                train_speed = train_speed * 10
#                train_op = tf.train.GradientDescentOptimizer(train_speed).minimize(loss_op)
#                print("Training speed is now: {}".format(train_speed))
#        else:
#            confirmed_count = 0
#            failed_count += 1
#            if failed_count > max_fail_count:
#                restore(saver, sess, "./{}".format(name))
#                train_speed = train_speed / 10
#                if train_speed <= 1e-7:
#                    break
#                failed_count = 0
#                print("Reverting to best model ({})".format(best_loss))
#                train_op = tf.train.GradientDescentOptimizer(train_speed).minimize(loss_op)
#                print("Training speed is now: {}".format(train_speed))
#
#        epoch_num = epoch_num + 1
#
#        if epoch_num > num_epochs:
#            print("Training finished!")
#            break
#
#    return best_loss
#

#def genetic_train(sess, generator):
#    num_models = 5
#    scale_factors = [1e2, 1, 1e-1, 1e-2, 1e-3]
#    model_losses = [np.inf] * 10
#
#    for m in range(num_models):
#        xs, output = build_model(scale_factors[m])
#        ys, loss_op = create_loss(output, WARMUP)
#
#        epoch_size = len(pliki) // BATCH_SIZE
#
#        model_losses[m] = train(sess, loss_op, xs, ys, generator, True, epoch_size, name=str(m), max_fail_count=5)
#
#    print("Best model: {}".format(np.argmin(model_losses)))
#    print("Loss: {}".format(np.min(model_losses)))
#
#TRAINING = True
#LOAD_WEIGHTS = True
#WARMUP = False
#
#
#with tf.Session() as sess:
#
#    #genetic_train(sess, generate_data)
#    xs, output = build_model()
#    ys, loss_op = create_loss(output, WARMUP)
#
#    epoch_size = len(pliki) // BATCH_SIZE
#
#    if TRAINING:
#        train(sess, loss_op, xs, ys, generate_data, not LOAD_WEIGHTS, epoch_size)
#    else:
#        saver = initialize(sess)
#        restore(saver, sess, "model")
#
#        cap = cv2.VideoCapture("/home/mipo57/Desktop/test.mp4")
#
#        while (cap.isOpened()):
#            ret, frame = cap.read()
#            #border = (frame.shape[0] - frame.shape[1]) // 2
#            #frame = cv2.copyMakeBorder( frame, 0, 0, border, border, cv2.BORDER_CONSTANT)
#            frame_resized = cv2.resize(frame, (416, 416))
#            test = cv2.blur(frame_resized, (3, 3)).astype(np.float32) / 255
#            prediction = sess.run(output, feed_dict={xs: np.reshape(test, (1, IMG_SIZE, IMG_SIZE, 3))})
#            prediction = np.reshape(prediction, (13, 13, 5))
#
#            cell = np.argmax(prediction[..., 4])
#            # true_cell = np.argmax(adnotacje[img_num, :, :, 4])
#
#            print(cell)
#            # print(true_cell)
#
#            prediction = np.reshape(prediction, ((13 * 13, 5)))
#
#            print(np.max(prediction[..., 4]))
#
#            x = (prediction[cell, 0] + cell % 13) * (frame.shape[1] // 13)
#            y = (prediction[cell, 1] + cell // 13) * (frame.shape[0] // 13)
#            w = prediction[cell, 2] * frame.shape[1]
#            h = prediction[cell, 3] * frame.shape[0]
#            if np.max(prediction[..., 4]) > 0.85:
#                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 255))
#
#            cv2.imshow('frame', frame)
#            if cv2.waitKey(21) & 0xFF == ord('q'):
#                break
#
# cap.release()
# cv2.destroyAllWindows()
#
#        while(True):
#            img_num = np.random.randint(0, len(pliki))
#            test = cv2.imread("/home/mipo57/Desktop/test.jpg")
#            test = cv2.blur(test, (3,3)).astype(np.float32) / 255
#            prediction = sess.run(output, feed_dict={xs: np.reshape(test, (1, IMG_SIZE, IMG_SIZE, 3))})
#            prediction = np.reshape(prediction, (13, 13, 5))
#
#            cell = np.argmax(prediction[..., 4])
#            #true_cell = np.argmax(adnotacje[img_num, :, :, 4])
#
#            print(cell)
#            #print(true_cell)
#
#            prediction = np.reshape(prediction, ((13*13, 5)))
#
#            print(np.max(prediction[..., 4]))
#
#            x = (prediction[cell, 0] + cell % 13) * (IMG_SIZE//13)
#            y = (prediction[cell, 1] + cell // 13) * (IMG_SIZE//13)
#            w = prediction[cell, 2] * IMG_SIZE
#            h = prediction[cell, 3] * IMG_SIZE
#
#            cv2.rectangle(test, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 255))
#            cv2.imshow('img', test)
#            cv2.waitKey(0)
#