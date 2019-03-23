#!/bin/sh
if ! test -f /home/ubuntu/caffe/examples/MobileNet-SSD/example_voc/solver_train.prototxt ;then
	echo "error: mobilenetv2_voc/solver_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
../../build/tools/caffe train -solver="/home/ubuntu/caffe/examples/MobileNet-SSD/example_voc//solver_train.prototxt" \
			      -gpu 0 \
			      -weights="/home/ubuntu/caffe/examples/MobileNet-SSD/snapshot/voc_21_origin/mobilenet_iter_55000.caffemodel"
#-weights="snapshot_mobilenet_8classes/mobilenet_iter_107000.caffemodel"
#-snapshot="/home/ubuntu/caffe/examples/MobileNet-SSD/snapshot_mobilenet_8classes/mobilenet_iter_107000.solverstate"






