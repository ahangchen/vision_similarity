import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
import argparse

def extract_feature(img_path, net):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = net.predict(x)
    return feature


def get_similarity(img_path1, img_path2):
    net = load_model('market_pair_pretrain.h5')
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    f1 = extract_feature(img_path1, net)
    f2 = extract_feature(img_path2, net)
    f1 = f1/np.linalg.norm(f1)
    f2 = f1/np.linalg.norm(f2)
    sim = np.sum(f1*f2)
    return sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute similarity of two images, example:\n python vision_sim.py --img1 /home/cwh/0.jpg --img2 /home/cwh/2.jpg')
    parser.add_argument('--img1', help='path of image1')
    parser.add_argument('--img2', help='path of image2')
    args = parser.parse_args()
    print get_similarity(args.img1, args.img2)