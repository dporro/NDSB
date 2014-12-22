#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize 
from skimage.filter import threshold_adaptive, inverse
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import mahotas
from mahotas import zernike, moments, features
from mahotas import lbp
import pdb
import datetime
import sys
sys.path.append('./xgboost-master/wrapper')
import xgboost as xgb
import pickle
# make graphics inline
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

##data_path = 'E:\Competitions\NationalDataScienceBowl'
data_path = '/home/sandrovegapons/Documents/Competitions/NDSB'
#
# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join(data_path,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(data_path,"train","*.*")))))

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop
#
## Rescale the images and create the combined metrics and training labels
    
def get_mahotas_features(image, bin_image):
    """
    """    
#    pdb.set_trace()
    zer = zernike.zernike_moments(image, len(image)/2)  #adding 25
    mm = []
    for i in range(7):
        for j in range(7):
            mm.append(np.log10(moments(image, i,j)))  #adding 49
    rou = features.roundness(bin_image)        #adding 1
    el1 = features.ellipse_axes(bin_image)[0]
    el2 = features.ellipse_axes(bin_image)[0]#adding 2
    exe = features.eccentricity(bin_image)    #adding 1
    pft = features.pftas(image)               #adding 54    
    hara = mahotas.features.haralick(image).reshape(-1)   #adding 52   
#    return np.concatenate((zer, np.array([rou]), np.array(ellip), hara.reshape(-1), lbps.reshape(-1)))
    #adding in total 184 features
    return np.concatenate((zer, np.array(mm), np.array([rou, el1, el2, exe]), pft, hara))

#
def get_max_region_features(max_region):
    """
    """
    feats = []
    feats.append(max_region.area)
    feats.append(max_region.convex_area)
    feats.append(max_region.centroid[0])
    feats.append(max_region.centroid[1])
    feats.append(max_region.eccentricity)
    feats.append(max_region.equivalent_diameter)
    feats.append(max_region.euler_number)
    feats.append(max_region.extent)
    feats.append(max_region.filled_area)
    feats.append(max_region.inertia_tensor_eigvals[0])
    feats.append(max_region.inertia_tensor_eigvals[1])
    feats += max_region.inertia_tensor.reshape(-1).tolist() #add 4 values
    feats.append(max_region.local_centroid[0])
    feats.append(max_region.local_centroid[1])
    feats.append(max_region.major_axis_length)
    feats.append(max_region.minor_axis_length)
    feats.append(max_region.orientation)
    feats.append(max_region.perimeter)
    feats.append(max_region.solidity)
    feats += max_region.moments.reshape(-1).tolist() #add 16 values
    feats += max_region.moments_central.reshape(-1).tolist() #add 16 values
    feats += max_region.moments_hu.reshape(-1).tolist() #add 7 values
    #we are computing 61 features
    
    feats += max_region.weighted_moments.reshape(-1).tolist() #add 16 values
    feats += max_region.weighted_moments_central.reshape(-1).tolist() #add 16 values
    feats += max_region.weighted_moments_hu.reshape(-1).tolist() #add 7 values
    #adding 39 features
    
    #adding 100 features
    return np.array(feats)
#
#
##get the total training images
#numberofImages = 0
#for folder in directory_names:
#    for fileNameDir in os.walk(folder):   
#        for fileName in fileNameDir[2]:
#             # Only read in the images
#            if fileName[-4:] != ".jpg":
#              continue
#            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 48
imageSize = maxPixel * maxPixel
#num_rows = numberofImages # one row for each image in the training dataset
num_features = 100 + 184
piv = 100

## X is the feature vector with one row of features per image
## consisting of the pixel values and our metric
#X = np.zeros((num_rows, num_features), dtype=float)
## y is the numeric class label 
#y = np.zeros((num_rows))
#
#files = []
## Generate training data
#i = 0    
#label = 0
# List of string of class names
namesClasses = list()
#cls = dict()

#print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
#    for fileNameDir in os.walk(folder):   
#        for fileName in fileNameDir[2]:
#            # Only read in the images
#            if fileName[-4:] != ".jpg":
#              continue
#            
#            # Read in the images and create the features
#            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
#            image = imread(nameFileImage, as_grey=True)
#            files.append(nameFileImage)
#            
#            image = image.copy()
##            image = resize(image, (maxPixel, maxPixel))
#            
#            # Create the thresholded image to eliminate some of the background
#            imagethr = np.where(image > np.mean(image),0.,1.0)        
#            #Dilate the image
#            imdilated = morphology.dilation(imagethr, np.ones((4,4)))        
#            # Create the label list
#            label_list = measure.label(imdilated)
#            label_list = imagethr*(label_list+1) #+1 to avoid the case when the region of interest takes label 0
#            label_list = label_list.astype(int)            
#            region_list = measure.regionprops(label_list, intensity_image=image)
#            maxregion = getLargestRegion(region_list, label_list, imagethr)
#             
#            X[i,:piv] = get_max_region_features(maxregion)
#            X[i, piv:] = get_mahotas_features(image,imdilated)
##            X[i, imageSize+1:] = get_mahotas_features(image, imdilated)
#            
#            # Store the classlabel
#            y[i] = label
#            i += 1
#            # report progress for each 5% done  
#            report = [int((j+1)*num_rows/20.) for j in range(20)]
#            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
#    label += 1
    

print "Training"

#pickle.dump(X, open('X', 'wb'))
#pickle.dump(y, open('y', 'wb'))

X = pickle.load(open('X', 'rb'))
print X.shape
y = pickle.load(open('y', 'rb'))



def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
    
    
#def compute_weights(class_report='class_report.txt', y=None):
#    """
#    """
#    w = np.zeros(y.shape[0])
#    data_path = '/home/sandrovegapons/Documents/Competitions/NDSB'
#    clwg = dict()
#    with open(os.path.join(data_path, class_report), 'rb') as rep:
#        lines = rep.readlines()
#        lines = lines[2:-2]
#        for l in lines:
#            vals = [v for v in l.strip().split(' ') if not v == '']
#            key = os.path.split(vals[0])[-1]
#            prec = vals[1]
#            reca = vals[2]
#            f1sc = vals[3]
#            supp = vals[4]
#            val = 1 + (int(supp))/1000.
#            clwg[key] = val
#    for i,v in enumerate(y):
##        pdb.set_trace()
#        k = os.path.split(namesClasses[int(v)])[-1]
#        w[i] = clwg[k]
#        
#    return w    
        
                  

param = {}    
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.06
param['max_depth'] = 17
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 121

num_round = 350

#w = compute_weights(y=y)
#pdb.set_trace()

#kf = KFold(y, n_folds=5)
## prediction probabilities number of samples, by number of classes
#y_pred = np.zeros((len(y),len(set(y))))
#for train, test in kf:
##    X_train, X_test, y_train, y_test, w_train, w_test = X[train,:], X[test,:], y[train], y[test], w[train], w[test]
#    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
##    xg_train = xgb.DMatrix(X_train, label=y_train, weight=w_train)
#    xg_train = xgb.DMatrix(X_train, label=y_train)
#
#    xg_test = xgb.DMatrix(X_test, label=y_test)
#    watchlist = [(xg_train,'train'), (xg_test, 'test')]
#
#    bst = xgb.train(param, xg_train, num_round, watchlist )
#    # get prediction
#    y_pred[test] = bst.predict(xg_test).reshape(y_test.shape[0], 121)
#
#    print 'iter'
#    
#print multiclass_log_loss(y, y_pred)

#print classification_report(y, y_pred, target_names=namesClasses)
    
#training with the whole training set
print 'Training with the whole training set'

xg_train = xgb.DMatrix(X, label=y)
watchlist = [(xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist)


##Reading the test set
        
print 'Reading test data'
header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')       
labels = map(lambda s: s.split('/')[-1], namesClasses)            
#get the total test images
fnames = glob.glob(os.path.join(data_path, "test", "*.jpg"))
numberofTestImages = len(fnames)
print numberofTestImages
#X_test = np.zeros((numberofTestImages, num_features), dtype=float)
images = map(lambda fileName: fileName.split('/')[-1], fnames)

#i = 0
## report progress for each 5% done  
#report = [int((j+1)*numberofTestImages/20.) for j in range(20)]
#for fileName in fnames:
#    # Read in the images and create the features
##    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
#    image = imread(fileName, as_grey=True)
#    
#    image = image.copy()
##    image = resize(image, (maxPixel, maxPixel))
#    
#    # Create the thresholded image to eliminate some of the background
#    imagethr = np.where(image > np.mean(image),0.,1.0)        
#    #Dilate the image
#    imdilated = morphology.dilation(imagethr, np.ones((4,4)))        
#    # Create the label list
#    label_list = measure.label(imdilated)
#    label_list = imagethr*(label_list+1) #+1 to avoid the case when the region of interest takes label 0
#    label_list = label_list.astype(int)            
#    region_list = measure.regionprops(label_list, intensity_image=image)
#    maxregion = getLargestRegion(region_list, label_list, imagethr)
#     
#    X_test[i,:piv] = get_max_region_features(maxregion)
#    X_test[i, piv:] = get_mahotas_features(image, imdilated)
# 
#    i += 1
#    if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"
    

#X_test = pickle.dump(X_test, open('X_test', 'wb'))
X_test = pickle.load(open('X_test', 'rb'))

xg_test = xgb.DMatrix(X_test)

y_pred = bst.predict(xg_test).reshape(X_test.shape[0], 121)
print 'writing the submission file'
df = pd.DataFrame(y_pred, columns=labels, index=images)
df.index.name = 'image'
df = df[header]
df.to_csv(os.path.join(data_path, "sub.csv"))
#!gzip competition_data/submission.csv



