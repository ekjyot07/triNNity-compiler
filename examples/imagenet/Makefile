.PHONY: clean

ILSVRC2012_img_val.tar:
	wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

ILSVRC2012_img_val: ILSVRC2012_img_val.tar
	mkdir -p ILSVRC2012_img_val && cd ILSVRC2012_img_val && tar --restrict --checkpoint -xf ../ILSVRC2012_img_val.tar && cd -

ILSVRC2012_devkit_t12.tar.gz:
	wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz

ILSVRC2012_devkit_t12: ILSVRC2012_devkit_t12.tar.gz
	tar -xzf ILSVRC2012_devkit_t12.tar.gz

%.caffemodel:
	wget http://dl.caffe.berkeleyvision.org/bvlc_$(shell echo $* | tr A-Z a-z).caffemodel -O $*.caffemodel

validate-%: ILSVRC2012_img_val ILSVRC2012_devkit_t12
	python validate.py --model $* \
                     --weights $*.caffemodel \
                     --classes imagenet-classes.txt \
                     --ground-truth=ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt \
                     --validation-set=ILSVRC2012_img_val \
