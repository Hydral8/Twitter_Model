# NOTE REQUIRES USER TO CLONE TENSORFLOW MODELS GITHUB REPOSITORY AND INSTALL OBJECT DETECTION PACKAGE
# BY DOING cd /models/research and then pip install .

# NOTE for this script to work, you
# must change /models/research/object_detection/utils/label_map_util.py so that
# tf.gfile is not used but tf.io.gfile (in TF 2.0 tf.gfile -> tf.io.gfile)


# also requires additional dependencies (TENSORFLOW) and would help to install TPU (tensorflow GPU) for accelerated development

# imports
import tensorflow as tf
import os
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import pandas as pd
# from PIL import Image
import argparse
from collections import namedtuple

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv", dest="input_csv",
                    help="CSV used to create TF records")
parser.add_argument("--image_dir", dest="image_dir",
                    help="Image dir used to get image sizes")
parser.add_argument("--label_map_path", dest="label_map_path",
                    help="Labels mappings to assign integer values to text labels")
parser.add_argument("--output_path", dest="output_path",
                    help="Output Path for TF records")
FLAGS = parser.parse_args()


def create_tf_record(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_png = fid.read()
    # image = Image.open(os.path.join(path, group.filename))
    (width, height) = (group.object.iloc[0].width, group.object.iloc[0].height)

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/image_format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))

    return tf_record


def split(df):
    data = namedtuple("data", ['filename', 'object'])
    gb = df.groupby("filename")
    return [data(filename, gb.get_group(group_data)) for filename, group_data in zip(gb.groups.keys(), gb.groups)]


def main():
    print(FLAGS.output_path)
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    # read dataset in
    df = pd.read_csv(FLAGS.input_csv)
    groups = split(df)
    for group in groups:
        record = create_tf_record(group, path, label_map_dict)
        writer.write(record.SerializeToString())

    writer.close()
    print("Successfully created tf records file to this path: {}".format(
        FLAGS.output_path))


main()
