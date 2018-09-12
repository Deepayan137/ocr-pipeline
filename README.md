# ocr-pipeline

## Requirements

```
pytorch 0.3 or above
python3.x
```

## Directory structure

```
ocr-pipeline
	|-- ocr.py
	|--dir1
		|--Images
		|--Segmentations
	|--dir2
		|--Images
		|--Segmentations
```

**Note**: The segmentation should be in the format of a text file containing
the co-ordinates of bounding box for each line in 4 seperate columns (x_top, y_top, width, height)


## parameters

```
-p path to directory containing images and Segmentations
-l language of the script you want to recognize
```

**Note:** the current pipeline supports recognition of only Hindi and Telugu scripts.

## To run the code

```
python ocr.py batch_ocr -p path/to/directory -l language/
```

The above code will create a folder, Prediction inside which there will be recognized text in the form of text files with names corresponding to the image name.

## pretrained models

You will need to download pretrained model.
you can download them from [here](https://drive.google.com/open?id=1e4ukpAewCqmAK7eb6vuBxlM6uXXywV5b)
unzip them and store it in a directory named models inside your ocr-pipeline directory.