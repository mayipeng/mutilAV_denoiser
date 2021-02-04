import glob
import random
import os

basedir = "/home2/mayipeng/myp_mutilAV/data/audio/u/drspeech/data/TCDTIMIT/Noisy_TCDTIMIT/"
cleanBasedir = "/home2/mayipeng/myp_mutilAV/data/audio/Clean/volunteers/"
filenames = glob.glob(basedir + '*/'+ '5'+'/volunteers/' +'*/'+'*/'+'*.wav')

if __name__=='__main__':
    index = 0
    total_file = len(filenames)
    print(total_file)

    random.shuffle(filenames)

    while index < total_file:
        filename = filenames[index]

        print("dirtyFilename: " + filename)
        print("cleanFilename: " + os.path.join(cleanBasedir,filename.split('/')[14], filename.split('/')[15],filename.split('/')[16]))
        index = index + 1
