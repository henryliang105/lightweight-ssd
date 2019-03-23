import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/ubuntu/caffe/'
sys.path.insert(0, caffe_root + 'python')

from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString

import caffe  
import time
import glob
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


net_file= 'example_sep/MobileNetSSD_deploy.prototxt'  
caffe_model='snapshot/mobilenet_iter_30000.caffemodel'
#caffe_model='snapshot/mobilenet_iter_107000.caffemodel' # not merge_bn

#net_file= 'MobileNetv2SSDLite_deploy.prototxt'  
#caffe_model='MobileNetv2SSDLite_deploy.caffemodel'

test_dir = "images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  


CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

"""
CLASSES = ['background',
           'bike', 'bus', 'car', 'motorbike', 'person', 'truck', 'van']
"""

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    cv2.imshow("preprocess", img)
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)


imgs_path = sorted(glob.glob('/home/ubuntu/caffe/examples/MobileNet-SSD/images/*.jpg'))

fontScale = 1
thickness = 3

def detect(ori_img):
    img = preprocess(ori_img)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(ori_img, out)

    for i in range(len(box)):
        #p1 = (box[i][0], box[i][1])
        #p2 = (box[i][2], box[i][3])
        xmin = box[i][0]
        ymin = box[i][1]
        xmax = box[i][2]
        ymax = box[i][3]
        #cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,255,0))
        p3 = (max(xmin, 15), max(ymin, 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])

        """
        color_picker = {
        'person' : (255, 0, 0),
        'car' : (0, 255, 0),
        'motorbike' : (0, 0, 255),
        'bus' : (0, 255, 255),
        'bike' : (255, 255, 0),
        'truck' : (128, 255, 0),
        'van' : (255, 128, 0)
        }

        color = color_picker[CLASSES[int(cls[i])]]
        """

        cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(ori_img, title, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,255,0), thickness, cv2.LINE_AA)
    return ori_img

def parse_xml():
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'GTSDB'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = '000001.jpg'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '500'

    node_height = SubElement(node_size, 'height')
    node_height.text = '375'

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'mouse'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '99'
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '358'
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '135'
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '375'

    xml = tostring(node_root, pretty_print=True) 
    dom = parseString(xml)
    print xml


for img_path in imgs_path:
    print(img_path)
    img = cv2.imread(img_path)

    #parse_xml(img_path)

    timer = Timer()
    timer.tic()
    dst_img = detect(img)
    timer.toc()
    img_name = img_path.split('/')[-1]
    cv2.putText(dst_img, img_name, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(dst_img, "FPS : " + str(int(1 / timer.total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("result", dst_img)

    print("detection time :", timer.total_time)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
