import tensorflow as tf
import re
import string
import json

from tensorflow.keras import layers
from tensorflow.keras import losses

def loadModel():
	model = tf.keras.models.load_model('/model/sentiment.h5')
	return model

def loadVectorizeLayerModel():
	vectorizeLayerModel = tf.keras.models.load_model('/model/vectorize_layer_model')
	return vectorizeLayerModel

def predict(review):
	model = loadModel() 

	max_features = 10000
	sequence_length = 250

	vectorize_layer = loadVectorizeLayerModel().layers[0]
	
	export_model = tf.keras.Sequential([
		vectorize_layer,
		model,
		layers.Activation('sigmoid')
	])

	export_model.compile(
		loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
	)
	predictions = export_model.predict(review)
	predictionsReturn = []

	for i, prediction in enumerate(predictions):
		p = {}
		p['key'] = i
		if prediction >= 0.5:
			p['prediction'] = "Positive"
		else:
			p['prediction'] = "Negative"
		predictionsReturn.append(p) 

	return json.dumps({'movieReviews':predictionsReturn})


if __name__ == "__main__":
	with tf.device("/cpu:0"):
		examples = [
		  "The movie was great!",
		  "The movie was okay.",
		  "The movie was terrible..."
		]

		print(predict(examples))





