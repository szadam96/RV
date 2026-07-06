import os
import pydicom
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import click


@click.command("preprocess")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
@click.option("--crop_size", type=(int, int), default=(112,112))
def run(data_dir, output, crop_size):
    """
    Preprocesses videos in the data_dir and saves them to the output directory.
    """
    dcm_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_paths.append(Path(os.path.join(root, file)))

    output = Path(output)
    os.makedirs(output, exist_ok=True)

    for dcm_path in tqdm(dcm_paths):
        if not os.path.exists(output / (dcm_path.stem + ".dcm.avi")):
            try:
                makeVideo(str(dcm_path), output, crop_size)
            except Exception as e:
                print(f"Error processing {dcm_path}: {e}")
                
def mask(output):
    dimension = output.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    

    mask = ((m1+m2)>int(dimension/2) + int(dimension/10)) 
    mask *=  ((m1-m2)<int(dimension/2) + int(dimension/10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)
    
    #print(maskedImage.shape)
    
    return maskedImage

def makeVideo(fileToProcess, destinationFolder, cropSize = (256,256), flip=False):
    try:
        fileName = fileToProcess.split('/')[-1] #\\ if windows, / if on mac or sherlock
                                                 #hex(abs(hash(fileToProcess.split('/')[-1]))).upper()

        if not os.path.isdir(os.path.join(destinationFolder,fileName)):

            dataset = pydicom.dcmread(fileToProcess, force=True)
            testarray = dataset.pixel_array
            if len(testarray.shape) == 3:
                testarray = np.stack([testarray, testarray, testarray], axis=3)

            frame0 = testarray[0]
            mean = np.mean(frame0, axis=1)
            mean = np.mean(mean, axis=1)
            try:
                yCrop = np.where(mean<1)[0][0]
            except:
                yCrop = 0
            testarray = testarray[:, yCrop:, :, :]

            bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
            if bias>0:
                if testarray.shape[1] < testarray.shape[2]:
                    testarray = testarray[:, :, bias:-bias, :]
                else:
                    testarray = testarray[:, bias:-bias, :, :]


            #print(testarray.shape)
            frames,height,width,channels = testarray.shape

            fps = 30

            try:
                fps = dataset[(0x18, 0x40)].value
            except:
                print("couldn't find frame rate, default to 30")

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            video_filename = os.path.join(destinationFolder, fileName + '.avi')
            out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)


            for i in range(frames):

                outputA = testarray[i,:,:,0]
                smallOutput = outputA[int(height/10):(height - int(height/10)), int(height/10):(height - int(height/10))]

                # Resize image
                output = cv2.resize(smallOutput, cropSize, interpolation = cv2.INTER_CUBIC)

                finaloutput = mask(output)
                if flip:
                    #flip the video horizontally
                    finaloutput = np.flip(finaloutput, axis=1)

                finaloutput = cv2.merge([finaloutput,finaloutput,finaloutput])
                out.write(finaloutput)

            out.release()

        else:
            print(fileName,"hasAlreadyBeenProcessed")
    except Exception as e:
        raise e
    return 0

