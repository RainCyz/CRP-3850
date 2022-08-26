# This program is used to calculate the green view index based on the collecte metadata. The
# Object based images classification algorithm is used to classify the greenery from the GSV imgs
# in this code, the meanshift algorithm implemented by pymeanshift was used to segment image
# first, based on the segmented image, we further use the Otsu's method to find threshold from
# ExG image to extract the greenery pixels.

# For more details about the object based image classification algorithm
# check: Li et al., 2016, Who lives in greener neighborhoods? the distribution of street greenery and it association with residents' socioeconomic conditions in Hartford, Connectictu, USA

# This program implementing OTSU algorithm to chose the threshold automatically
# For more details about the OTSU algorithm and python implmentation
# cite: http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html


# Copyright(C) Xiaojiang Li, Ian Seiferling, Marwa Abdulhai, Senseable City Lab, MIT
# First version June 18, 2014

# last modified by Yucheng Zhang, only to fetch GSV images

# using 18 directions is too time consuming, therefore, here I only use 6 horizontal directions
# Each time the function will read a text, with 1000 records, and save the result as a single TXT
def GSV_Fetch(GSVinfoFolder, outTXTRoot, greenmonth, key_file):

    """
    This function is used to download the GSV from the information provide
    by the gsv info txt, and save the result to a shapefile

    Required modules: StringIO, numpy, requests, and PIL

        GSVinfoTxt: the input folder name of GSV info txt
        outTXTRoot: the output folder to store result green result in txt files
        greenmonth: a list of the green season, for example in Boston, greenmonth = ['05','06','07','08','09']
        key_file: the API keys in txt file, each key is one row, I prepared five keys, you can replace by your owne keys if you have Google Account

    last modified by Xiaojiang Li, MIT Senseable City Lab, March 25, 2018

    """

    import time
    from PIL import Image
    import numpy as np
    import requests
    from StringIO import StringIO


    # read the Google Street View API key files, you can also replace these keys by your own
    lines = open(key_file,"r")
    keylist = []
    for line in lines:
        key = line[:-1]
        keylist.append(key)

    print ('The key list is:=============', keylist)

    # set a series of heading angle
    headingArr = 360/6*np.array([0,1,2,3,4,5])
    print(headingArr)

    # number of GSV images for Green View calculation, in my original Green View View paper, I used 18 images, in this case, 6 images at different horizontal directions should be good.
    numGSVImg = len(headingArr)*1.0
    pitch = 0

    # create a folder for GSV images and grenView Info
    if not os.path.exists(outTXTRoot):
        os.makedirs(outTXTRoot)

    # the input GSV info should be in a folder
    if not os.path.isdir(GSVinfoFolder):
        print 'You should input a folder for GSV metadata'
        return
    else:
        allTxtFiles = os.listdir(GSVinfoFolder)
        for txtfile in allTxtFiles:
            if not txtfile.endswith('.txt'):
                continue

            txtfilename = os.path.join(GSVinfoFolder,txtfile)
            lines = open(txtfilename,"r")

            # create empty lists, to store the information of panos,and remove duplicates
            panoIDLst = []
            panoDateLst = []
            panoLonLst = []
            panoLatLst = []

            # loop all lines in the txt files
            for line in lines:
                metadata = line.split(" ")
                panoID = metadata[1]
                panoDate = metadata[3]
                month = panoDate[-2:]
                lon = metadata[5]
                lat = metadata[7][:-1]

                # print (lon, lat, month, panoID, panoDate)

                # in case, the longitude and latitude are invalide
                if len(lon)<3:
                    continue

                # only use the months of green seasons
                if month not in greenmonth:
                    continue
                else:
                    panoIDLst.append(panoID)
                    panoDateLst.append(panoDate)
                    panoLonLst.append(lon)
                    panoLatLst.append(lat)

            # the output text file to store the green view and pano info
            gvTxt = 'GV_'+os.path.basename(txtfile)
            GreenViewTxtFile = os.path.join(outTXTRoot,gvTxt)

            # Consider aggregate the Scene_Parsing part here?
            # Seems to be less effective, as the other thread waits for the
            # Image Classification

            # check whether the file already generated, if yes, skip. Therefore, you can run several process at same time using this code.
            print GreenViewTxtFile
            if os.path.exists(GreenViewTxtFile):
                continue

            # write the green view and pano info to txt
            with open(GreenViewTxtFile,"w") as gvResTxt:
                for i in range(len(panoIDLst)):
                    panoDate = panoDateLst[i]
                    panoID = panoIDLst[i]
                    lat = panoLatLst[i]
                    lon = panoLonLst[i]

                    # get a different key from the key list each time
                    #idx = i % len(keylist)
                    #key = keylist[idx]

                    for heading in headingArr:
                        print "Heading is: ",heading

                        # using different keys for different process, each key can only request 25,000 imgs every 24 hours
                        URL = "http://maps.googleapis.com/maps/api/streetview?size=400x400&pano=%s&fov=60&heading=%d&pitch=%d&sensor=false&key=AIzaSyCHMs8Als3x2oQaaHuvJdtMSUjtOUvPX9A"%(panoID,heading,pitch)
                        #print(URL)
                        # let the code to pause by 1s, in order to not go over data limitation of Google quota
                        #time.slee(1)
                        time.sleep(0.2) #in this instance, sleep less as a cornell AAP student

                        # classify the GSV images and calcuate the GVI
                        try:
                            response = requests.get(URL)
                            im = np.array(Image.open(StringIO(response.content)))

                            """
                            Download the image for review?
                            """
                            dld_file = "D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\FULL\\image\\" + str(panoID) + str(heading) + ".png"
                            #print(dld_file)
                            with open(dld_file,'wb') as f:
                                f.write(response.content)


# ------------------------------Main function-------------------------------
if __name__ == "__main__":

    import os,os.path
    import itertools


    GSVinfoRoot = 'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\FULL\\pano_txt'
    outputTextPath = r'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\FULL\\GVI_output_text'
    greenmonth = ['05','06','07','08','09']
    key_file = 'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\Treepedia\\keys.txt'

    GSV_Fetch(GSVinfoRoot,outputTextPath, greenmonth, key_file)
