FILE=$1

if [[ $FILE != "apple2orange" && $FILE != "horse2zebra" && $FILE != "maps" ]]; then
    echo "Available datasets are: apple2orange, horse2zebra, maps, cezanne2photo, monet2photo"
    exit 1
fi


echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
