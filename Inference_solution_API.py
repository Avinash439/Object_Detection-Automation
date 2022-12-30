import os
import numpy as np
import cv2
import pandas as pd
import json
import base64
import tensorflow as tf
import time
import pathlib
import matplotlib.pyplot as plt
import warnings
import urllib.request
from io import BytesIO
import uuid
import requests

from flask import Flask, jsonify, make_response, request
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import BadRequest
from PIL import Image

from collections import Counter
warnings.filterwarnings('ignore') 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



flask_app = Flask(__name__,static_url_path='/static')

## efficient Model.


api = Api(app = flask_app, version = '1.0', title = 'Object_Model(80)', description = 'Localizing objects in an image')
multi_label_resources = api.model('GridCountResources',
                                          {
                                            'userId': fields.String(required = True, default = ' ', description="Input userId", help="input_path"),
                                            'transactionId': fields.String(required = True, default = '', description = 'transactionid'),
                                            'image': fields.String(required = True, default = ' ', description="Image base64 string value", help="image can not be null"),
                                            'Threshold':  fields.Float(required = True, default = 0.5, description="Image base64 string value", help="image can not be null")
                                            # 'file_path': fields.String(required = True, default = ' ', description = 'Image path')
                                          })
                                              


def filter_predictions(pred_dict:dict, threshold:int):
    
    detection_boxes = pred_dict['detection_boxes'].tolist()
    detection_scores = pred_dict['detection_scores'].tolist()
    class_names = pred_dict['detection_classes'].tolist()

    # filterd_detection_classes_indices = [index for index, class_name in enumerate(class_names) if len(class_names) >=3]
    filterd_detection_scores_indices = [index for index, score in enumerate(detection_scores) if score >= threshold]

    
    pred_dict['detection_boxes'] = [detection_boxes[index] for index in filterd_detection_scores_indices]
    pred_dict['detection_classes'] = [class_names[index] for index in filterd_detection_scores_indices]
    pred_dict['detection_scores'] = [detection_scores[index] for index in filterd_detection_scores_indices]

    
    

    return pred_dict


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

#end point
@api.route('/Object_Detection')
class ASL_multiLabelClassification(Resource):


  def get_normal_image_from_bytes(self, img_bytes):
    jpg_original = base64.b64decode(img_bytes)
    img_as_np = np.fromstring(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(img_as_np, flags=1)
    return img
    
  def Load_model(self,userId,transactionId):

    PATH_TO_SAVED_MODEL = "/home/ob_reserved_cu/scripts/static/store_data/" +userId +"/" + transactionId +"/saved_model/saved_model"# + userId+"_"+"Fasterrcnn" + "_"+ "custom_classes" + "_" + transactionId+ "_trained_model_v1"+ ".zip"
    PATH_TO_LABELS= "/home/ob_reserved_cu/scripts/static/store_data/" +userId +"/" + transactionId + "/"+userId+"_"+"Fasterrcnn"+"_"+"custom_classes"+"_"+transactionId+"_trained_model_v1"+".pbtxt"

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)  

    return detect_fn,category_index     
                                                                   
  def verify_token(self,headers):
        verify_token=headers['token']
        #    ip=headers['ip']
        ip="10.7.246.9" 
        valid_users=requests.post(f"http://{ip}/NslIotHubAPI/authenticateToken?token={verify_token}").json()
        if valid_users['message']=='Token Valid':
            return True
        else:
            return False



  @api.expect(multi_label_resources)
  def post(self):

    output={'detection_boxes':[],'detection_scores':[],'class_names':[], 'summary':'','image_url':''}
    # output={'detection_boxes':[],'detection_scores':[],'class_names':[], 'summary':''}
    ids=[]
    names=[]
    class_names=[]
    data = request.get_json()
    headers=request.headers
    if not self.verify_token(headers):
      return {"message":"Token not valid"}
    img_link = data['image']
    userId = data['userId']
    transactionId = data['transactionId']

    # Read the input Image for classification
    with urllib.request.urlopen(img_link) as url:
        img = tf.keras.preprocessing.image.load_img(BytesIO(url.read()), target_size=(640, 640))
        img_array = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')

        
        # img_array = tf.expand_dims(img_array, 0)

    threshold=data['Threshold']
    # img_path = data['file_path']
    # image_np=self.get_normal_image_from_bytes(img_link) 
    # img_saved_path="static/images/testimage.jpg"
    # cv2.imwrite(img_saved_path,image_np)
    # output['image_url']='http://164.52.213.64:5025/'+img_saved_path

    print('Example Inference code on a single image')

    # image_np = load_image_into_numpy_array(image)

    input_tensor = tf.convert_to_tensor(img_array)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[:, :, :, :3] 


   
    detect_fn,category_index=self.Load_model(userId,transactionId)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    print(detections)
    thresholded_dic=filter_predictions(detections,threshold)               
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = list(np.array(detections['detection_classes']).astype(np.int64))
    # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    output['detection_boxes']=thresholded_dic['detection_boxes']
    
    classes=thresholded_dic['detection_classes']
    output['detection_scores']=thresholded_dic['detection_scores']
    # output['detection_boxes']=detections['detection_boxes'].tolist()
    # classes=detections['detection_classes'].tolist()
    # output['detection_scores']=detections['detection_scores'].tolist()
    
    print("detections",output)
    print(category_index)
    
##########################################################################
    for key in category_index:
        #print("first key is :", key)
        #print(type(one[key]))
        for value in category_index[key]:
            # print("the dict of dict id is :", i)
            if value=='id':
                ids.append(category_index[key][value])
            # print(ids)
            if value=='name':
                names.append(category_index[key][value])
            # print(name)


    dictionary = dict(zip(ids, names))    

    print(dictionary) 

    
    for item in classes:
      class_names.append(dictionary[item])

    output['class_names']=class_names  
    dict_classnames = dict(Counter(output['class_names']))
    li = []
    for key, value in dict_classnames.items():
      li.append(key +'('+str(value)+')')
    output['summary'] = ', '.join(li)

    image_np_with_detections = img_array.copy()
      
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          np.array(detections['detection_boxes']),
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=threshold,
          agnostic_mode=False)

    # plt.figure()

    # plt.imshow(image_np_with_detections)
    img_saved_path="/home/ob_reserved_cu/scripts/static/store_data/" + userId +"/"+ transactionId + "/sol_inference/"
    
    if not os.path.exists(img_saved_path):
        os.makedirs(img_saved_path)

    img_path = '%s/%s.jpg' % (img_saved_path,  str(uuid.uuid4()))
    cv2.imwrite(img_path,cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))
    basename=(os.path.basename(img_path))


    output['image_url']="http://164.52.213.64:5336/static/store_data/"+ userId +"/"+ transactionId + "/sol_inference/" + basename

    print('Done')
# plt.show()
  

    
    return jsonify(output)



if __name__ == '__main__':

    flask_app.run(host='0.0.0.0', port=5336, debug=True, use_reloader=False, threaded=True)                                              