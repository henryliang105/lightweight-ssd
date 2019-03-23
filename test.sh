#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/mobilenetv1_voc/mobilenetv1_iter_53000.caffemodel | head -n 4)
if test -z $latest; then
	exit 1
fi
../../build/tools/caffe train -solver="example_voc/solver_test.prototxt" \
--weights=$latest \
-gpu 0
