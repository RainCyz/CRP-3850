#This is a PyTorch implementation of semantic segmentation models on MIT ADE20K
#scene parsing dataset (http://sceneparsing.csail.mit.edu/).

# An experimental usage of this code snippet can be found at:
# https://colab.research.google.com/drive/1tgtE9aBbA_9QvqpNN6Nzzqix_ifZG3z9?usp=sharing

%%bash
# Colab-specific setup
!(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit
pip install yacs 2>&1 >> install.log
git init 2>&1 >> install.log
git remote add origin https://github.com/CSAILVision/semantic-segmentation-pytorch.git 2>> install.log
git pull origin master 2>&1 >> install.log
DOWNLOAD_ONLY=1 ./demo_test.sh 2>> install.log

# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    display(PIL.Image.fromarray(im_vis))


# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

def seg(img):
    pil_image = PIL.Image.open(img).convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    with torch.no_grad():
      scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred
    #=====================Visualization=======================================#
    #visualize_result(img_original, pred)


    # Top classes in answer
    #predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
    #for c in predicted_classes[:10]:
        #visualize_result(img_original, pred, c)

#----------------------------------------------------------------------------#
def GVI_calculation(img):
    resolution = 400*400
    veg_code = {
        'tree' : 4,
        'grass' : 9,
        'palm' : 72}
    pred = seg(img)
    pixel_count = 0
    for i in greenIndex.values():
        pixel_count += len(np.where(pred == i)[0]) # accumulate pixel count
    GVI_Pct = pixel_count/resolution*100
    return GVI_Pct

def GreenViewComputing_segpars(GSVinfoFolder, outTXTRoot, greenmonth, key_file):

    """
    This function is used to download the GSV from the information provide
    by the gsv info txt, and save the result to a shapefile

    Required modules: StringIO, numpy, requests, and PIL

        GSVinfoTxt: the input folder name of GSV info txt
        outTXTRoot: the output folder to store result green result in txt files
        greenmonth: a list of the green season, for example in Boston, greenmonth = ['05','06','07','08','09']
        key_file: the API keys in txt file, each key is one row, I prepared five keys, you can replace by your owne keys if you have Google Account

    modified by Xiaojiang Li, MIT Senseable City Lab, March 25, 2018

    last modified by Yucheng Zhang, to incorporate the ML based sceneparsing algo
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

                    # calculate the green view index
                    greenPercent = 0.0

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

                            dld_file = "D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\sampleStrImg\\" + str(panoID) + str(heading) + ".png"
                            #print(dld_file)
                            with open(dld_file,'wb') as f:
                                f.write(response.content)

                        # if the GSV images are not download successfully or failed to run, then return a null value
                        except Exception as e:
                            print(e)
                            greenPercent = -1000
                            break

                    greenViewVal = seg(im)
                    print 'The greenview: %s, pano: %s, (%s, %s)'%(greenViewVal, panoID, lat, lon)

                    # write the result and the pano info to the result txt file
                    lineTxt = 'panoID: %s panoDate: %s longitude: %s latitude: %s, greenview: %s\n'%(panoID, panoDate, lon, lat, greenViewVal)
                    gvResTxt.write(lineTxt)

#====================================Main=====================================#
if __name__ == "__main__":
    import os
    from os import *

    GSVinfoRoot = 'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\FULL\\multi_threading\\pano_2'
    outputTextPath = r'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\FULL\\GVI_output_text'
    greenmonth = ['05','06','07','08','09']
    key_file = 'D:\\2022 Spring\\CRP 3850\\CRP-3850\\Green View Index\\Treepedia\\keys.txt'

    GreenViewComputing_segpars(GSVinfoFolder, outTXTRoot, greenmonth, key_file)
"""
    imgs = os.listdir(GSVFOLDER) #fill in your folder
    for path in imgs:
        img = path
        GVI = GVI_calculation(img)
        print 'The greenview: %s'%(GVI)
"""
