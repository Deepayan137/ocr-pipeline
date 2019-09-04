# ocr-pipeline

## Requirements

```
pytorch 0.3 or above
python3.x
warp-ctc https://github.com/SeanNaren/warp-ctc
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
--path path to directory containing images and Segmentations
--lang language of the script you want to recognize
```

**Note:** the current pipeline supports recognition of only Hindi and Telugu scripts.

## To run the code

```
python ocr_multib.py batch_ocr --path=path/to/directory --lang=language/ --save_file=/path/to/saved/model 
```

The above code will create a folder, Predictions_CRNN inside which there will be recognized text in the form of text files with names corresponding to the image name.

## pretrained models

You will need to download pretrained model.
you can download them from [here](https://drive.google.com/open?id=1e4ukpAewCqmAK7eb6vuBxlM6uXXywV5b)
unzip them and store it in a directory named models inside your ocr-pipeline directory.
