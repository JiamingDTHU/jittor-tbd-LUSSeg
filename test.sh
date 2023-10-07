DATA=../ImageNet-S/ImageNetS50
IMAGENETS=../ImageNet-S/ImageNetS50

DUMP_PATH=./weights/pass50
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
ARCH=resnet18
NUM_CLASSES=50

echo "START INFERENCE: inference.py"

python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode test \
