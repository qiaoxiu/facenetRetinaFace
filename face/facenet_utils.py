from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
import time
import torch

TARGET_DIR = 'imgs/'

files = [f for f in os.listdir(TARGET_DIR)]
npz = np.load('all2.npz')
#global known_face_encodings, known_face_names, sids
known_face_encodings = npz['encode']
known_face_names = npz['names']
sids = npz['sids']
workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def _load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _get_all_encoding():
    global known_face_encodings, known_face_names, sids
    if os.path.exists('new_data2.npz'):
        new_npz = np.load('new_data2.npz')
        new_encodings = new_npz['encode']
        new_names = new_npz['names']
        new_sids = new_npz['sids']
        num = known_face_encodings.shape[0]
        new_num = new_encodings.shape[0]
        flag = 0
        for j in range(new_num):
            for i in range(num):
                if str(sids[i])==new_sids[j]:
                    #known_face_encodings[i] = new_encodings[:,np.newaxis].T
                    known_face_encodings[i] = new_encodings[j]
                    known_face_names[i] = new_names[0]
                    sids[i] = new_sids[0]
                    flag = 1
                    break
        if flag==0:
            #known_face_encodings = np.vstack((known_face_encodings, new_encodings[:,np.newaxis].T))
            known_face_encodings = np.vstack((known_face_encodings, new_encodings))
            sids = np.hstack((sids, new_sids))
            known_face_names = np.hstack((known_face_names, new_names))
        os.remove('new_data2.npz')
    print(known_face_names)
    np.savez('all2.npz', encode=known_face_encodings, sids=sids, names=known_face_names)
    return known_face_encodings, sids, known_face_names


def _face_distance(known_face_encoding_list, face_encoding_to_check):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(known_face_encoding_list) == 0:
        return np.empty((0))
    # return (face_encodings - face_to_compare).norm().item()
    # encodes_ = known_face_encoding_list - face_encoding_to_check[:, np.newaxis]
    return np.linalg.norm(known_face_encoding_list - face_encoding_to_check, axis=1)


def _compare_faces(known_face_encoding_list, face_encoding_to_check, tolerance=0.8):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(_face_distance(known_face_encoding_list, face_encoding_to_check) <= tolerance)


def recognition_name(img="lwx.jpg"):
    start = time.time()
    known_face_encodings, _, known_face_names =  _get_all_encoding()
    unknown_image = _load_image_file(img)
    unknown_face_encodings = face_encoding(img)
    pil_image = Image.fromarray(unknown_image)
    name = "Unknown"
    draw = ImageDraw.Draw(pil_image)
    matches = _compare_faces(known_face_encodings, unknown_face_encodings, tolerance=0.8)
    face_distances = _face_distance(known_face_encodings, unknown_face_encodings)
    print('face_distances: ', face_distances)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    print('name:',name)
    draw.text((100, 120), str(name), fill=(255, 255, 255, 255))

    end = time.time()
    print('耗时:', end-start)
    t_d = 'static/assets/img'
    up_path = os.path.join(t_d, 'aaa.jpg')
    pil_image.save(up_path, 'jpeg')
    return up_path


def _face_encodings(obj_img):
    x_aligned, prob = mtcnn(obj_img, return_prob=True)
    if x_aligned is None:
        return []
    aligned = torch.stack([x_aligned]).to(device)
    face_encodings = resnet(aligned).detach().cpu()
    return face_encodings[0].numpy()


def face_encoding(img):
    obj_img = _load_image_file(img)
    obj_face_encoding = _face_encodings(obj_img)
    return obj_face_encoding


def identification_face(img="lwx.jpg"):
    known_face_encodings, _, known_face_names = _get_all_encoding()
    start = time.time()
    face_encodings = face_encoding(img)
    name = "Unknown"
    matches = _compare_faces(known_face_encodings, face_encodings, tolerance=0.8)
    face_distances = _face_distance(known_face_encodings, face_encodings)
    print('face_distances: ', face_distances)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    end = time.time()
    print('耗时:', end-start)
    return name
recognition_name(img='imgs/liuwenxiu.jpg')
