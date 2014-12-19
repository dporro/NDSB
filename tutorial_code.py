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
from mahotas import zernike
from mahotas import lbp
import pdb
import datetime
# make graphics inline
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

#data_path = 'E:\Competitions\NationalDataScienceBowl'
data_path = '/home/sandrovegapons/Documents/Competitions/NDSB'

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join(data_path,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(data_path,"train","*.*")))))
 
## Example image
## This example was chosen for because it has two noncontinguous pieces
## that will make the segmentation example more illustrative
#example_file = glob.glob(os.path.join(directory_names[11],"*.jpg"))[9]
#print example_file
#im = imread(example_file, as_grey=True)
#plt.imshow(im, cmap=cm.gray)
#plt.show()
#
## First we threshold the image by only taking values greater than the mean to reduce noise in the image
## to use later as a mask
#f = plt.figure(figsize=(12,3))
#imthr = im.copy()
#imthr = np.where(im > np.mean(im),0.,1.0)
##imthr = np.abs(1-threshold_adaptive(im, block_size=len(im)/4))
#sub1 = plt.subplot(1,4,1)
#plt.imshow(im, cmap=cm.gray)
#sub1.set_title("Original Image")
#
#sub2 = plt.subplot(1,4,2)
#plt.imshow(imthr, cmap=cm.gray_r)
#sub2.set_title("Thresholded Image")
#
#imdilated = morphology.dilation(imthr, np.ones((3,3)))
#sub3 = plt.subplot(1, 4, 3)
#plt.imshow(imdilated, cmap=cm.gray_r)
#sub3.set_title("Dilated Image")
#
#
#
#labels = measure.label(imdilated)
#labels = imthr*labels
#labels = labels.astype(int)
#sub4 = plt.subplot(1, 4, 4)
#sub4.set_title("Labeled Image")
#plt.imshow(labels)

 
#pdb.set_trace()
## calculate common region properties for each region within the segmentation
#regions = measure.regionprops(labels)

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
  
    
#regionmax = getLargestRegion(props=regions, labelmap=labels, imagethres=imthr)
#plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
#plt.show()

#print regionmax.minor_axis_length/regionmax.major_axis_length

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio, label_list
    
# Rescale the images and create the combined metrics and training labels
    
def get_mahotas_features(image, bin_image):
    """
    """
    
#    pdb.set_trace()
    zer = zernike.zernike_moments(image, len(image)/2)
    rou = mahotas.features.roundness(bin_image)
    ellip = mahotas.features.ellipse_axes(bin_image)
#    hara = mahotas.features.haralick(bin_image)
#    lbps = lbp.lbp_transform(image, radius=1, points=8)
    
#    pdb.set_trace()
#    return np.concatenate((zer, np.array([rou]), np.array(ellip), hara.reshape(-1), lbps.reshape(-1)))
    return np.concatenate((zer, np.array([rou]), np.array(ellip)))


def get_max_region_features(max_region):
    """
    """
    feats = []
    feats.append(max_region.area)
    feats.append(max_region.convex_area)
    feats.append(max_region.centroid[0])
    feats.append(max_region.centroid[1])
    feats.append(maxregion.eccentricity)
    feats.append(maxregion.equivalent_diameter)
    feats.append(maxregion.euler_number)
    feats.append(maxregion.extent)
    feats.append(maxregion.filled_area)
    feats.append(maxregion.inertia_tensor_eigvals[0])
    feats.append(maxregion.inertia_tensor_eigvals[1])
    feats += maxregion.inertia_tensor.reshape(-1).tolist() #add 4 values
    feats.append(max_region.local_centroid[0])
    feats.append(max_region.local_centroid[1])
    feats.append(maxregion.major_axis_length)
    feats.append(maxregion.minor_axis_length)
    feats.append(maxregion.orientation)
    feats.append(maxregion.perimeter)
    feats.append(maxregion.solidity)
    feats += maxregion.moments.reshape(-1).tolist() #add 16 values
    feats += maxregion.moments_central.reshape(-1).tolist() #add 16 values
    feats += maxregion.moments_hu.reshape(-1).tolist() #add 7 values
    #we are computing 61 features
    return np.array(feats)


#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 45
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
#num_features = imageSize + 1 + maxPixel + 1 + 2  # for our ratio
num_features = 61 + 25 + 1 + 2

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()
cls = dict()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            
            image = image.copy()
#            image = resize(image, (maxPixel, maxPixel))
            
            # Create the thresholded image to eliminate some of the background
            imagethr = np.where(image > np.mean(image),0.,1.0)        
            #Dilate the image
            imdilated = morphology.dilation(imagethr, np.ones((3,3)))        
            # Create the label list
            label_list = measure.label(imdilated)
            label_list = imagethr*(label_list+1) #+1 to avoid the case when the region of interest takes label 0
            label_list = label_list.astype(int)            
            region_list = measure.regionprops(label_list)
            maxregion = getLargestRegion(region_list, label_list, imagethr)
             
            X[i,:61] = get_max_region_features(maxregion)
            X[i, 61:] = get_mahotas_features(image,imdilated)
#            X[i, imageSize+1:] = get_mahotas_features(image, imdilated)
            
            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1
    
# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

##Create a DataFrame object to make subsetting the data on the class 
#df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})
#
#f = plt.figure(figsize=(30, 20))
##we suppress zeros and choose a few large classes to better highlight the distributions.
#df = df.loc[df["ratio"] > 0]
#minimumSize = 20 
#counts = df["class"].value_counts()
#largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
## Loop through 40 of the classes 
#for j in range(0,40,2):
#    subfig = plt.subplot(4, 5, j/2 +1)
#    # Plot the normalized histograms for two classes
#    classind1 = largeclasses[j]
#    classind2 = largeclasses[j+1]
#    n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
#                         alpha=0.5, bins=[x*0.01 for x in range(100)], \
#                         label=namesClasses[classind1].split(os.sep)[-1], normed=1)
#
#    n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
#                          alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
#    subfig.set_ylim([0.,10.])
#    plt.legend(loc='upper right')
#    plt.xlabel("Width/Length Ratio")


print "Training"
## n_estimators is the number of decision trees
## max_features also known as m_try is set to the default value of the square root of the number of features
#clf = RF(n_estimators=100, n_jobs=3);
#scores = cross_validation.cross_val_score(clf, X, y, cv=KFold(y, n_folds=5), n_jobs=1);
#print "Accuracy of all classes"
#print np.mean(scores)

#kf = KFold(y, n_folds=5, shuffle=True)
#y_pred = y * 0
#for train, test in kf:
#    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
#    clf = RF(n_estimators=100, n_jobs=3)
#    clf.fit(X_train, y_train)
#    y_pred[test] = clf.predict(X_test)
#print classification_report(y, y_pred, target_names=namesClasses)


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
    

# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y),len(set(y))))
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    clf = RF(n_estimators=600, n_jobs=5)
#    clf = GradientBoostingClassifier(n_estimators=100, max_depth=6)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict_proba(X_test)
    print 'iter'
    
print multiclass_log_loss(y, y_pred)

#print classification_report(y, y_pred, target_names=namesClasses)
    
#training with the whole training set
print 'Training with the whole training set'
clf = RF(n_estimators=600, n_jobs=5)
clf.fit(X, y)

##Reading the test set
#print 'Reading test data'
#test_path = 'E:\\Competitions\\NationalDataScienceBowl\\test'
#
#with open('sub.csv', 'wb') as submission:
#    submission.write('image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified\n')
#    for i,f in enumerate(os.listdir(test_path)):
#        if i>0 and i%10000==0:
#            print 'iteration: %s, time: %s' %(i, datetime.datetime.now())
#        x = np.zeros(imageSize + 1 + maxPixel + 1 + 2 )
#        nameFileImage = os.path.join(test_path, f)
#        image = imread(nameFileImage, as_grey=True)
#        axisratio = getMinorMajorRatio(image)
#        image = resize(image, (maxPixel, maxPixel))
#        
#        imagethr = np.where(image > np.mean(image),0.,1.0)
#        imdilated = morphology.dilation(imagethr, np.ones((4,4)))
#        
#        # Store the rescaled image pixels and the axis ratio
#        x[0:imageSize] = np.reshape(image, (1, imageSize))
#        x[imageSize] = axisratio
#        x[imageSize+1:] = get_mahotas_features(imdilated)
#        
#        pred = clf.predict_proba(x)[0]
#        wr = f
#        for v in pred:
#            wr += ','+str(v)
#        wr += '\n'
##        wr = f + ',' + str(pred)[1:-1].replace(' ','') + '\n'
#        submission.write(wr)
        
print 'Reading test data'
header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')       
labels = map(lambda s: s.split('/')[-1], namesClasses)            
#get the total test images
fnames = glob.glob(os.path.join(data_path, "test", "*.jpg"))
numberofTestImages = len(fnames)
print numberofTestImages
X_test = np.zeros((numberofTestImages, num_features), dtype=float)
images = map(lambda fileName: fileName.split('/')[-1], fnames)

i = 0
# report progress for each 5% done  
report = [int((j+1)*numberofTestImages/20.) for j in range(20)]
for fileName in fnames:
    # Read in the images and create the features
#    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
    image = imread(fileName, as_grey=True)
    
    image = image.copy()
#    image = resize(image, (maxPixel, maxPixel))
    
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)        
    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((5,5)))        
    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*(label_list+1) #+1 to avoid the case when the region of interest takes label 0
    label_list = label_list.astype(int)            
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
     
    X_test[i,:61] = get_max_region_features(maxregion)
    X_test[i, 61:] = get_mahotas_features(image, imdilated)
 
    i += 1
    if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"

y_pred = clf.predict_proba(X_test)
print 'writing the submission file'
df = pd.DataFrame(y_pred, columns=labels, index=images)
df.index.name = 'image'
df = df[header]
df.to_csv(os.path.join(data_path, "sub.csv"))
#!gzip competition_data/submission.csv



