# Core libraries
import os
import io
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 
from sklearn import metrics

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable
import torchvision
from torchvision import transforms

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50

# Import our dataset class
from datasets_copy.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

import cv2

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

# python test.py --model_path=output/fold_0/best_model_state.pkl --folds_file=datasets_copy/OpenSetCows2020/splits/10-90-custom.json --save_path=output/fold_0/

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"  # Set the GPU 2 to use

# For a trained model, let's evaluate it
def evaluateModel(args):
	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, True)
	test_dataset = Utilities.selectDataset(args, False)
	# print(len(test_dataset))

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels= inferEmbeddings(args, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, test_dataset, "test")

	# print(train_embeddings)
	# print(test_embeddings)
	# Classify them
	# accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
	accuracy, pred= KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
	print(pred)
	# print(test_labels)
	# Write it out to the console so that subprocess can pick them up and close
 
	# print(f"Accuracy={round(accuracy, 3)}")
	sys.stdout.write(f"Accuracy={str(accuracy)}")
	sys.stdout.flush()
	sys.exit(0)


# test.py 만으로도 기존 학습된 모델로 특징점 추출은 가능함
# 리턴 받은 특징점 값들을 신규 등록할때 신규 라벨을 부여하면서 따로 저장하고 
# 그 다음 데이터 확인할때 기존 데이터 먼저 확인 후 예외처리를 통해 신규 라벨 데이터를 확인하면 추가적인 모델 학습 없이도 가능할것 같다?
# 단점은 특징점 추출값이 1개이기 때문에 같은 소를 다른 각도에서 찍은 사진으로 확인할때 판별을 못할 수 있다?
# 처음 등록할때 사진을 여러장을 찍어서 등록한다?

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels ,n_neighbors=17):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)
    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels-1)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)
    # print(test_embeddings)
    p1 = predictions.astype(int)
    # print(predictions)
    # print(f'test: {predictions[0]}')
    
    conf_list = []
    fail_list = []
    vic_list =[]
    c_count = 0
    
    print(p1)
    print(f'test: {p1[0]}')
    
    # for i in range(len(p1)):
    #     label = train_labels[p1[i]]
    #     y_predict = neigh.predict_proba(test_embeddings)
    #     # print(y_predict)
    #     confidence = y_predict[i][y_predict[i].argmax()]
    #     # print(confidence)
    #     if p1[i] == test_labels[i]:
    #         vic = 'correct'
    #         vic_list.append(round(confidence, 3))
    #         if confidence >= 0.7:
    #             c_count += 1
    #     else:
    #         vic = 'failed'
    #         fail_list.append(f'{p1[i]} : {round(confidence, 3)} : {vic}')
    #     # conf_list.append(f'{p1[i]} : {round(confidence, 3)} : {vic}')
    
    # print(conf_list)
    # print(fail_list)
    # print(min(vic_list))
    
    # print(test_labels)
    
    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100
    accuracy2 = (float(c_count) / total) * 100
    
    print(f'======================        {accuracy}        ======================')
    # print(f'======================        {accuracy2}        ======================')
    
    return accuracy, predictions

# # Use KNN to classify the embedding space
# def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels):
# 	r_list = []
# 	for k in range(1, 101):
# 		# Define the KNN classifier
# 		neigh = KNeighborsClassifier(n_neighbors=k, n_jobs=-4)
# 		# print(train_labels, test_labels)
# 		# Give it the embeddings and labels of the training set
# 		neigh.fit(train_embeddings, train_labels)

# 		# Total number of testing instances
# 		total = len(test_labels-1)
# 		# Get the predictions from KNN
# 		predictions = neigh.predict(test_embeddings)
# 		# print(test_labels.shape, predictions.shape)
# 		# print(test_embeddings)
# 		p1 = predictions.astype(int)
# 		# print(predictions)
# 		# print(f'test: {predictions[0]}')
  
# 		ta = metrics.accuracy_score(test_labels, predictions)

# 		conf_list = []
# 		fail_list = []
# 		vic_list =[]
# 		c_count = 0

# 		# print(p1)
# 		# print(f'test: {p1[0]}')

# 		for i in range(len(p1)):
# 			label = train_labels[p1[i]]
# 			y_predict = neigh.predict_proba(test_embeddings)
# 			# print(y_predict)
# 			confidence = y_predict[i][y_predict[i].argmax()]
# 			# print(confidence)
# 			if p1[i] == test_labels[i]:
# 				vic = 'correct'
# 				vic_list.append(round(confidence, 3))
# 				if confidence >= 0.7:
# 					c_count += 1
# 			else:
# 				vic = 'failed'
# 				fail_list.append(f'{p1[i]} : {round(confidence, 3)} : {vic}')

# 			# How many were correct?
# 			correct = (predictions == test_labels).sum()

# 		# Compute accuracy
# 		accuracy = (float(c_count) / total) * 100

# 		print(f'======================    {k} test:       {accuracy}        ======================')
# 		# print(f'======================    {i} result:       {ta}        ======================')
		
# 		r_list.append(accuracy)
# 		a = max(r_list)
# 		b = r_list.index(a)
# 		# print(r_list)
# 	print(f'max : {a}, neighbor : {b+1}')
# 	return accuracy, predictions


# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataset, split):
	# Wrap up the dataset in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Define our embeddings model
	model = resnet50(pretrained=True, num_classes=dataset.getNumClasses(), ckpt_path=args.model_path, embedding_size=args.embedding_size)
	# print(dataset.getNumClasses())
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()
	
	# timage = np.array(cv2.imread('/workspace/bum/MetricLearningIdentification/datasets_copy/OpenSetCows2020/images/test/002/2.jpg'))
	# timage = cv2.resize(timage, dsize=(214, 214),interpolation=cv2.INTER_LINEAR)
	# timage_swap = np.swapaxes(timage, 0,2)
	# timage_swap = np.expand_dims(timage_swap, axis=0)
 
	# test_tensor = torch.from_numpy(timage_swap).type(torch.cuda.FloatTensor)
	# test_result = model(test_tensor)
	# print(model(test_result))
	# print(model(test_result).shape)
	
 
	# Embeddings/labels to be stored on the testing set
	outputs_embedding = np.zeros((1,args.embedding_size))
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

 
	# Iterate through the testing portion of the dataset and get
	for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
		# Put the images on the GPU and express them as PyTorch variables
		# print(type(images))
		# print(images.shape)
  
		images = Variable(images.cuda())
		
  		# Get the embeddings of this batch of images
		outputs = model(images)

		# print(images.shape)
		# print(outputs)
		# if co == 0:
		# 	break
		
		# Express embeddings in numpy form
		embeddings = outputs.data
		embeddings = embeddings.cpu().numpy()

		# Convert labels to readable numpy form
		labels = labels.view(len(labels))
		# print(labels)
		labels = labels.cpu().numpy()
		# print(labels)

		# Store testing data on this batch ready to be evaluated
		outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
		labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
		# print(outputs_embedding)
		# print(labels_embedding)
	
	# If we're supposed to be saving the embeddings and labels to file
	if args.save_embeddings:
		# Construct the save path
		save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
		
		# Save the embeddings to a numpy array
		np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

	return outputs_embedding, labels_embedding

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, required=True, 
						help='Path to the saved model to load weights from')
	parser.add_argument('--folds_file', type=str, default="", required=True,
						help="The file containing known/unknown splits")
	parser.add_argument('--save_path', type=str, required=True,
						help="Where to store the embeddings")

	parser.add_argument('--dataset', nargs='?', type=str, default='OpenSetCows2020', 
						help='Which dataset to use')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
	args = parser.parse_args()


	# print(args.model_path)
	# print(args.folds_file)
	# print(args.save_path)
	# Let's infer some embeddings
	evaluateModel(args)