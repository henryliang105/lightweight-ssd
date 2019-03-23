import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/ubuntu/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time
import kcftracker
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


net_file= 'example/8_class_haitech_prototxt/MobileNetSSD_deploy.prototxt'  
caffe_model='MobileNetSSD_8_classes_deploy.caffemodel' #nighttime model
#caffe_model='MobileNetSSD_8_classes_haitech_deploy.caffemodel' #daytime model
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ['background',
           'bike', 'bus', 'car', 'motorbike', 'person', 'truck', 'van']

#cap = cv2.VideoCapture('/home/ubuntu/Videos/haitech/test/day/test2.avi')
#cap = cv2.VideoCapture('/home/ubuntu/Videos/haitech/test/day/test6.avi')
#cap = cv2.VideoCapture('/home/ubuntu/Videos/haitech/test/night/test1.avi')
cap = cv2.VideoCapture('/home/ubuntu/Videos/haitech/test/night/test2.avi')



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

img_height = 300
img_width = 300

def preprocess(src):
    img = cv2.resize(src, (img_height, img_width))
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

trackers = []

def create_tracker(xmin, ymin, w, h, frame):
    tracker = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale
    tracker.init([xmin, ymin, w, h], frame)
    trackers.append(tracker)


def check_object(xmin, ymin, xmax, ymax, frame):
    is_duplicate = False
    del_trackers_idx = []
    update_tracker = -1

    
    if(len(trackers) == 0):
        create_tracker(xmin, ymin, (xmax - xmin), (ymax - ymin), frame)
    else:
        for i, tracker in enumerate(trackers):
            boundingbox, peak_value = tracker.update(frame)
            boundingbox = list(map(int, boundingbox))

            print((boundingbox[0],boundingbox[1]))

            cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0, 0, 255), 1)
            #print("boundingbox:", boundingbox[0][1])
            boxA = (xmin, ymin, xmax, ymax)
            boxB = (boundingbox[0], boundingbox[1], boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[3])
            
            if(IOU(boxA, boxB) > 0.4):
                is_duplicate = True
                update_tracker = i
                break

            if peak_value < 0.9:
                del_trackers_idx.append(i)

        if(is_duplicate == True and update_tracker > -1):
            del trackers[update_tracker]
            create_tracker(xmin, ymin, (xmax - xmin), ymax- ymin, frame)        
        elif (is_duplicate == False):
            create_tracker(xmin, ymin, (xmax - xmin), ymax- ymin, frame)
            
        del_trackers_idx.reverse()
        for i in del_trackers_idx:
            print('i: ', i)
            del trackers[i]


        print('len(trackers): ', len(trackers))




def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou




ori_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
ori_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

cap.set(cv2.CAP_PROP_POS_FRAMES, 6500)


fontScale = 0.8
thickness = 2


def detect(ori_img, framePos):
    img = preprocess(ori_img)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(ori_img, out)
    #print(framePos)
    for i in range(len(box)):
        #p1 = (box[i][0], box[i][1])
        #p2 = (box[i][2], box[i][3])
        xmin = box[i][0]
        ymin = box[i][1]
        xmax = box[i][2]
        ymax = box[i][3]
        #cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,255,0))
        p3 = (max(xmin, 15), max(ymin, 15))

        if conf[i] > 0.35  and ymin > 50 and xmin < 150:
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
            cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), color, 1)
            check_object(xmin, ymin, xmax, ymax, ori_img)
            cv2.putText(ori_img, title, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) #class label
    return ori_img


out = cv2.VideoWriter('/home/ubuntu/Videos/haitech/output/out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (img_width,img_height))
 

while(cap.isOpened()):
    ret, frame = cap.read()
    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    back_view = frame[5 : int(ori_height / 2 - 5), int(ori_width / 2 + 12) : int(ori_width) - 5]
    right_view = frame[int(ori_height / 2 + 5) : int(ori_height), 5 : int(ori_width / 2 - 12)]
    left_view = frame[int(ori_height / 2 + 5) : int(ori_height), int(ori_width / 2 + 12) : int(ori_width) - 5]



    #back views
    #back_view = cv2.resize(back_view, (img_width, img_height)) 
    right_view = cv2.resize(right_view, (img_width, img_height)) 
    #left_view = cv2.resize(left_view, (img_width, img_height)) 

    timer = Timer()
    timer.tic()
    print("framePos: ", framePos)
    detect(right_view, framePos)

    timer.toc()
    print ('Detection took {:.3f}s'.format(timer.total_time))
    #fps_file.write(str(int(1 / timer.total_time)) + '\n')
    #cv2.putText(back_view, "FPS : " + str(int(1 / timer.total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(right_view, "FPS : " + str(int(1 / timer.total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 0, 255), 2, cv2.LINE_AA)
    #cv2.putText(left_view, "FPS : " + str(int(1 / timer.total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 0, 255), 2, cv2.LINE_AA)
    #writer.write(ori_image)

    #cv2.imshow("left_view", back_view)
    cv2.imshow("right", right_view)
    #cv2.imshow("left", left_view)

    out.write(right_view)

    if cv2.waitKey(1) & 0xFF == ord('q') or framePos == 9000:
        break
cap.release()
cv2.destroyAllWindows()



