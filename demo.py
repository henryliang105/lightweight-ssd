import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/ubuntu/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


net_file= 'example/8_class_prototxt/MobileNetSSD_deploy.prototxt'  
caffe_model='example/8_class_prototxt/MobileNetSSD_8_classes_deploy.caffemodel'
#caffe_model='snapshot/mobilenet_iter_107000.caffemodel' # not merge_bn

#net_file= 'MobileNetv2SSDLite_deploy.prototxt'  
#caffe_model='MobileNetv2SSDLite_deploy.caffemodel'

test_dir = "images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
"""
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""

CLASSES = ['background',
           'bike', 'bus', 'car', 'motorbike', 'person', 'truck', 'van']


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
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)



#front
#cap = cv2.VideoCapture('/home/ubuntu/Videos/GoPro Hero 5 Session-night.mp4') #start:30000
#cap = cv2.VideoCapture('/home/ubuntu/Videos/YDXJ0341.mp4')
#cap = cv2.VideoCapture('/home/ubuntu/Videos/YDXJ0344.mp4')
#cap = cv2.VideoCapture('/home/ubuntu/Videos/YDXJ0349.mp4')
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/front/front2.mp4')


#back
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/back/back3.mp4')
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/back/back2.mp4') #start:40000

#left
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/left/left9.mp4') #start:0
cap = cv2.VideoCapture('/home/ubuntu/Videos/left14.mp4')
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/left/left15.mp4')

#right
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/right/right5.mp4')
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/right/right6.mp4')

cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)
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
        

        cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(ori_img, title, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
    return ori_img


while(cap.isOpened()):
    ret, frame = cap.read()
    src_frame = cv2.resize(frame, (500, 375))
    frame = cv2.resize(frame, (960, 540))
    
    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timer = Timer()
    timer.tic()
    dst_img = detect(frame)
    timer.toc()
    cv2.putText(dst_img, "FPS : " + str(int(1 / timer.total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("result", dst_img)
    cv2.imwrite("front2/src/"+ 'ft2_' + str(int(framePos)) + '.jpg', src_frame)
    cv2.imwrite('result_front2/' + 'ft2_' + str(int(framePos)) + '.jpg', dst_img)
    print("detection time :", timer.total_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#for f in os.listdir(test_dir):
#    if detect(test_dir + "/" + f) == False:
#       break
