import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from collections import Counter

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str, help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True, help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true', help='Ignore all images without annotations. Keep only these with at least one annotation')
parser.add_argument('--multi-class', dest='multi_class', action='store_true', help='Split a multi-class dataset while preserving class distributions in train and test sets')
args = parser.parse_args()

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info', {})
        licenses = coco.get('licenses', [])
        images = coco.get('images', [])
        annotations = coco.get('annotations', [])
        categories = coco.get('categories', [])

        if args.having_annotations:
            images_with_annotations = set([ann['image_id'] for ann in annotations])
            images = [img for img in images if img['id'] in images_with_annotations]

        if args.multi_class:
            annotation_categories = [ann['category_id'] for ann in annotations]

            # Remove classes that have only one sample because they can't be split into the training and testing sets
            category_counts = Counter(annotation_categories)
            valid_categories = {cat for cat, count in category_counts.items() if count > 1}

            annotations = [ann for ann in annotations if ann['category_id'] in valid_categories]

            # Create a mapping from category_id to category name
            category_id_to_name = {cat['id']: cat['name'] for cat in categories}

            X = np.array(annotations)
            y = np.array(annotation_categories)

            X_train, y_train, X_test, y_test = iterative_train_test_split(X.reshape(-1, 1), y.reshape(-1, 1), test_size=1 - args.split)

            X_train = X_train.flatten().tolist()
            X_test = X_test.flatten().tolist()

            train_annotations = X_train
            test_annotations = X_test

            train_images = filter_images(images, train_annotations)
            test_images = filter_images(images, test_annotations)

            save_coco(args.train, info, licenses, train_images, train_annotations, categories)
            save_coco(args.test, info, licenses, test_images, test_annotations, categories)

            print("Saved {} entries in {} and {} in {}".format(len(train_annotations), args.train, len(test_annotations), args.test))

        else:
            X_train, X_test = train_test_split(images, train_size=args.split)

            anns_train = filter_annotations(annotations, X_train)
            anns_test = filter_annotations(annotations, X_test)

            save_coco(args.train, info, licenses, X_train, anns_train, categories)
            save_coco(args.test, info, licenses, X_test, anns_test, categories)

            print("Saved {} entries in {} and {} in {}".format(len(anns_train), args.train, len(anns_test), args.test))

if __name__ == "__main__":
    main(args)
