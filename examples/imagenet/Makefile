.PHONY: clean

ILSVRC2012_img_val.tar:
	wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

ILSVRC2012_img_val: ILSVRC2012_img_val.tar
	tar -xf ILSVRC2012_img_val.tar

ILSVRC2012_devkit_t12.tar.gz:
	wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz

ILSVRC2012_devkit_t12: ILSVRC2012_devkit_t12.tar.gz
	tar -xzf ILSVRC2012_devkit_t12.tar.gz

validate: ILSVRC2012_img_val ILSVRC2012_devkit_t12
	python validate.py --model AlexNet \
                     -val_gt=ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
                     -data_dir=ILSVRC2012_img_val
