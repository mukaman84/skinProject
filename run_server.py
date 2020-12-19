import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
# from tensorflow as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
import albumentations as A
import segmentation_models as sm

from flask import Flask, render_template, request
from werkzeug import secure_filename

FILE_FOLDER = os.path.join('', '/static/upload_file/skin/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FILE_FOLDER

best_model_acne = None
best_graph_acne = None

best_model_hemo = None
best_graph_hemo = None

best_model_mela = None
best_graph_mela = None

best_model_pore = None
best_graph_pore = None

best_model_sebum = None
best_graph_sebum = None

best_model_wrinkle = None
best_graph_wrinkle = None

session = None

def load_model():
	global best_model_acne, best_graph_acne, best_model_hemo,best_graph_hemo, best_model_mela,best_graph_mela, best_model_pore,best_graph_pore, best_model_sebum, best_graph_sebum, best_model_wrinkle, best_graph_wrinkle, session

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)

	tf.keras.backend.set_session(session)

	BACKBONE = 'resnet101'
	BATCH_SIZE = 4
	CLASSES = ['acne']
	LR = 0.0001
	EPOCHS = 1000

	sm.set_framework('tf.keras')
	preprocess_input = sm.get_preprocessing(BACKBONE)

	# define network parameters
	n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
	activation = 'sigmoid' if n_classes == 1 else 'softmax'

	# create model
	#best_model_acne
	best_graph_acne = tf.get_default_graph()
	with best_graph_acne.as_default():
		best_model_acne = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_acne.load_weights('/home/skinai/ai/best_model_acne.h5')

	#best_model_hemo	
	best_graph_hemo = tf.get_default_graph()
	with best_graph_hemo.as_default():
		best_model_hemo = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_hemo.load_weights('/home/skinai/ai/best_model_hemo.h5')

	#best_model_mela
	best_graph_mela = tf.get_default_graph()
	with best_graph_mela.as_default():
		best_model_mela = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_mela.load_weights('/home/skinai/ai/best_model_mela.h5')

	#best_model_pore
	best_graph_pore = tf.get_default_graph()
	with best_graph_pore.as_default():
		best_model_pore = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_pore.load_weights('/home/skinai/ai/best_model_pore.h5')

	#best_model_sebum
	best_graph_sebum = tf.get_default_graph()
	with best_graph_sebum.as_default():
		best_model_sebum = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_sebum.load_weights('/home/skinai/ai/best_model_sebum.h5')

	#best_model_wrinkle
	best_graph_wrinkle = tf.get_default_graph()
	with best_graph_wrinkle.as_default():
		best_model_wrinkle = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
		best_model_wrinkle.load_weights('/home/skinai/ai/best_model_wrinkle.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skin')
def skin():
    return render_template('skin.html')

@app.route('/skinAi', methods = ['POST'])
def skinAi():
	if request.method  == 'POST':
		f = request.files['uploadfile']
		ai_target = request.form['ai_target']

		#저장경로에 파일 저장
		file_name = secure_filename(f.filename)
		f.save('/home/skinai/ai/static/upload_file/skin/' + ai_target + '/' + file_name)

		os.chmod('/home/skinai/ai/static/upload_file/skin/' + ai_target + '/' + file_name, 0o777)
		
		#AI 분석
		image = cv2.imread(os.path.join('/home/skinai/ai/static/upload_file/skin/' + ai_target + '/' + file_name))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = np.expand_dims(image, axis=0)

		if ai_target == 'acne':

			with best_graph_acne.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_acne.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)
			
		elif ai_target == 'hemo':

			with best_graph_hemo.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_hemo.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)
			
		elif ai_target == 'mela':

			with best_graph_mela.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_mela.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)
			
		elif ai_target == 'pore':

			with best_graph_pore.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_pore.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)

		elif ai_target == 'sebum':

			with best_graph_sebum.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_sebum.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)

		elif ai_target == 'wrinkle':

			with best_graph_wrinkle.as_default():
				tf.keras.backend.set_session(session)
				pr_mask = best_model_wrinkle.predict(image).round()
				pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),axis=2)
			
		output_path = os.path.join('/home/skinai/ai/static/upload_file/test/'+ai_target+'/') + file_name
		cv2.imwrite(output_path, pr_mask)
		os.chmod(output_path, 0o777)

		in_filename = os.path.join('/static/upload_file/skin/'+ai_target+'/') + file_name
		out_filename = os.path.join('/static/upload_file/test/'+ai_target+'/') + file_name

	return render_template('skin_ok.html',in_filename=in_filename,out_filename=out_filename,ai_target=ai_target)

if __name__ == "__main__":
	print(("* Loading AI model and Flask starting server..." "please wait until server has fully started"))
	load_model()
	app.run(host="0.0.0.0", port="6060")


#python3.6 run_server.py
#nohup python3.6 run_server.py &