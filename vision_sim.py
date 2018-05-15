
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
import argparse

def extract_feature(img_path, boxes, net):
#    img = image.load_img(img_path, target_size=(224, 224))
    img = image.load_img(img_path)
    feature = []
    for box in boxes:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        box = tuple(box)
        image_people = img.crop(box)
        image_people = image_people.resize((224, 224))
        x = image.img_to_array(image_people)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature.append(net.predict(x))
    return feature

def get_similarity(img_path1, img_path2, boxes_1, boxes_2):
    net = load_model('market_pair_pretrain.h5')
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    feature_1 = extract_feature(img_path1, boxes_1, net)
    feature_2 = extract_feature(img_path2, boxes_2, net)
    sim = [[0 for j in range(len(boxes_2))] for i in range(len(boxes_1))]
    for i in range(len(feature_1)):
        for j in range(len(feature_2)):
            f1 = feature_1[i]/np.linalg.norm(feature_1[i])
            f2 = feature_2[j]/np.linalg.norm(feature_2[j])
            sim[i][j] = np.sum(f1*f2)
    return sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute similarity of two images, example:\n python vision_sim.py --img1 /home/cwh/0.jpg --box1 [[382,193,73,126]] --img2 /home/cwh/2.jpg --box2 [[370,196,92,125]]')
    parser.add_argument('--img1', help='path of image1')
    parser.add_argument('--img2', help='path of image2')
    parser.add_argument('--box1', type=list, help='boxes of image1')
    parser.add_argument('--box2', type=list, help='boxes of image2')
    args = parser.parse_args()
    print(get_similarity(args.img1, args.img2, args.box1, args.box2))

