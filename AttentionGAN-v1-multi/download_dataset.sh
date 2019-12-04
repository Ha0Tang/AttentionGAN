FILE=$1

if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

fi