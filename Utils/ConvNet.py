import torch
from torch.autograd import Variable
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import requests


class ConvNet():
    def __init__(self, maxBoxes=50):
        """
        Input:
            maxBoxes - maximum number of boxes to retrieve
        """
        self.ConvNet = self.createModel()
        self.edgeBoxDetectionModel = self.createBoxDetectionModel()
        self.maxBoxes = maxBoxes
        


    def createModel(self):
        #CREATE MODEL   
        model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
        
        # Get first 6 layers, up to the 3rd convolutional layer
        modules=list(model.children())[0][:7]
        return torch.nn.Sequential(*modules)



    def createBoxDetectionModel(self):
        model_path = "model.yml.gz"

        if not os.path.isfile(model_path):

            url = 'https://github.com/markcNewell/ConvNet/blob/main/model.yml.gz?raw=true'
            r = requests.get(url, allow_redirects=True)

            open(model_path, 'wb').write(r.content)


        edge_detection_obj = cv2.ximgproc.createStructuredEdgeDetection(model_path)

        return edge_detection_obj


    def detectAndCompute(self, image, throw):
        return self.detectAndComputeSimple(image)

    def detectAndComputeSimple(self, image):
        """
        Input:
            image - OpenCV RGB image
        Output:
            ld_features - array of (n x 64896) image features
        """
        #REGION PROPOSALS    
        keypoints = self.edgeBoxDetection(image)
        
        #FEATURE DESCRIPTORS    
        features = np.array([]).reshape(-1, 64896)
        
        for box in keypoints:     
            x1,y1,x2,y2 = box
            
            img = image[y1:y2, x1:x2]

            #resize to 231 x 231 x 3
            img = cv2.resize(img, (231, 231), interpolation=cv2.INTER_AREA)

            #CONVERT NUMPY ARRAY TO TENSOR
            img = np.transpose(img,(2,0,1))
            img = torch.from_numpy(img)
            #rescale to be [0,1] like the data it was trained on by default 
            img = img.type('torch.DoubleTensor')
            img *= (1/255)
            #turn the tensor into a batch of size 1
            img = img.unsqueeze(0)
            img = img.type('torch.DoubleTensor')

            features_var = self.ConvNet(img.float()) # get the output from the last hidden layer of the pretrained resnet
            feature = features_var.data.numpy() # get the tensor out of the variable

            features = np.vstack([features, np.ravel(feature)])


        return keypoints, features



    def edgeBoxDetection(self, image):
        """
        Input:
            image - OpenCV RGB image
        Ouput:
            boxes - array of boxes where each box is (left, top, right, bottom)
        """
        # Get the edges
        edges = self.edgeBoxDetectionModel.detectEdges(np.float32(image)/255.0)
        # Create an orientation map
        orient_map = self.edgeBoxDetectionModel.computeOrientation(edges)
        # Suppress edges
        edges = self.edgeBoxDetectionModel.edgesNms(edges, orient_map)
        
        #Create edge box:
        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(self.maxBoxes)
        edge_boxes.setAlpha(0.5)
        edge_boxes.setBeta(0.5)
        prop_boxes, scores = edge_boxes.getBoundingBoxes(edges, orient_map)

        # Convert (x,y,w,h) parameters for the top 100 proposal boxes into (x, y, x+w, y+h) parameters
        # to be consistent with the xml tags of the ground truth boxes where (x,y) indicates the
        # top left corner and (x+w,y+h) indicates the bottom right corner of bounding box
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in prop_boxes]
        
        return boxes    



def normalize(arr):
    length = np.sqrt((arr**2).sum(axis=1))[:,None]
    return arr / length



def dimentionalityReduction(features, new_dimentions=512):
    #512, 1024, 4096 - different options for new dimentionality size
    transformer = GaussianRandomProjection(new_dimentions)
    ld_features = transformer.fit_transform(features)
    return ld_features, transformer



def matchFeatures(test_descriptors, test_keypoints, train_descriptors, train_keypoints, lowerDimentionality=512):
    """
    Input:
        test_descriptors - array of (n x 64896) test features
        test_keypoints - array of boxes where each box is (left, top, right, bottom)
        train_descriptors - array of (n x 64896) training features
        train_keypoints - array of boxes where each box is (left, top, right, bottom)
        lowerDimentionality - different options for new dimentionality size either {512, 1024, 4096}
    Output:
        S - similarities of the two sets of matches
        matches - the OpenCV Match Classes for the matches betwen the two datasets
    """
    #512, 1024, 4096 - different options for new dimentionality size
    transformer = GaussianRandomProjection(lowerDimentionality)
    ld_features = transformer.fit_transform(np.vstack([test_descriptors, train_descriptors]))

    # Normalize
    test = normalize(test_descriptors)
    train = normalize(train_descriptors)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(np.float32(test), np.float32(train), k=1)
    
    
    #SHAPE SIMILARITIES
    S = 0
    for match in matches:
        match = match[0]

        index_i = match.queryIdx
        x1,y1,x2,y2 = test_keypoints[index_i]
        wi = x2-x1
        hi = y2-y1

        index_j = match.trainIdx
        x1,y1,x2,y2 = train_keypoints[index_j]
        wj = x2-x1
        hj = y2-y1

        S += 1 - (match.distance * np.exp(0.5 * (((wi-wj)/np.max([wi,wj])) + ((hi-hj)/np.max([hi,hj])))))

    S /= np.sqrt(len(test) * len(train))
    
    return S, matches



def drawMatches(image1, kp1, image2, kp2, matches):
    """
    Input:
        image1 - OpenCV RGB image
        kp1 - array of boxes where each box is (left, top, right, bottom) associated with image1
        image2 - OpenCV RGB image
        kp2 - array of boxes where each box is (left, top, right, bottom) associated with image2
        matches - array of OpenCV Match objects
    """
    img1 = image1.copy()
    img2 = image2.copy()

    for match in matches:
        color = list(np.random.random(size=3) * 256)
        
        x1,y1,x2,y2 = kp1[match[0].queryIdx]
        cv2.rectangle(img1, (x1, y1), (x2, y2), color, 2)

        x1,y1,x2,y2 = kp2[match[0].trainIdx]
        cv2.rectangle(img2, (x1, y1), (x2, y2), color, 2)


    fig, ax = plt.subplots(2,1, figsize=(15,15))
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    