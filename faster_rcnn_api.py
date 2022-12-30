from flask import Flask, jsonify, make_response, request
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import BadRequest
import pandas as pd
import time
import os
import json
from pathlib import Path
import uuid
import numpy as np
import matplotlib.pyplot as plt
import traceback
import random
from sklearn.model_selection import train_test_split
import funcy
import configparser
import create_coco_tf_record
import tensorflow as tf
import cv2
import requests
import shutil
from models.research.object_detection import model_main_tf2
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import config_util
from models.research.object_detection import model_lib_v2
from models.research.object_detection import exporter_main_v2
import threading 
import kafka_utils
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)





flask_app = Flask(__name__,static_url_path='/static')

api = Api(app = flask_app, version = '1.0', title = 'Faster_rcnn', description = 'Localizing objects in an image')
frcnn_resources = api.model('GridCountResources',
                                          {
                                            'userId': fields.String(required = True, default = ' ', description="Input userId", help="input_path"),
                                            'transactionId': fields.String(required = True, default = '', description = 'transactionid'),
                                            'total_steps': fields.Integer(required = True, default = 5000, description="Number of epochs", help="epochs"),
                                            'batch_size': fields.Integer(required = True, default = 4, description="Batch_size", help="epochs"),
                                            'Learning_rate': fields.Float(required = True, default = 0.04, description="learning_rate", help="epochs"),
                                            'input_dict':fields.String(required=True, default='',description="kafka_input")
                                            
                                            
                                          })
                                              

@api.route('/train_network')
class train_fastercnn(Resource):



  def save_coco(self,file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)


  def checkfile(self,userId,transactionId):
    
    path='/home/ob_reserved_cu/scripts/static/store_data/' + userId+"/"
    print("path",path)
  
    if os.path.isdir(path):
      print("hello")
      os.chdir(path +transactionId)
      annotation=os.getcwd()+'/trainable.json'
      print("annotation_path",annotation)
      return annotation
    

  def generate_pbtxt(self,annotation_path,userId,transactionId):

    saved_path_pbtxt='/home/ob_reserved_cu/scripts/static/store_data/'+userId +'/'+transactionId +'/'+userId+"_"+"Fasterrcnn"+"_"+"custom_classes"+"_"+transactionId+"_trained_model_v1"+'.pbtxt'
    f = open(annotation_path)
    data = json.load(f)
    l=[]
    for dictt in data['categories']:
        l.append(dictt['name'])

    for i in range(len(l)):
        a=dict()
        a["id"]=i+1
        a["name"]=l[i]
        string=str(a).replace(", ","\n")
        string=string.replace("'id'","  id")
        string=string.replace("'name'","  name")
        string=string.replace("{","")
        string=string.replace("}","")
        with open(saved_path_pbtxt, 'a') as f:
            f.write("item {")
            f.write("\n")
            f.write(string)
            f.write("\n")
            f.write("}")
            f.write("\n\n")
            f.close()
    return saved_path_pbtxt          



  def filter_annotations(self,annotations, images):
      image_ids = funcy.lmap(lambda i: int(i['id']), images)
      return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

  def split_fun(self,annotation_path,train_path,test_path):
      with open(annotation_path, 'rt', encoding='UTF-8') as annotations:
          coco = json.load(annotations)
          info = coco['info']
          licenses = coco['licenses']
          images = coco['images']
          annotations = coco['annotations']
          categories = coco['categories']

          number_of_images = len(images)

          images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

          # if args.having_annotations:
          #     images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

          x, y = train_test_split(images, train_size=0.80)
          
          self.save_coco(train_path, info, licenses, x, self.filter_annotations(annotations, x), categories)
          self.save_coco(test_path, info, licenses, y, self.filter_annotations(annotations, y), categories)
          
          with open('split_status.txt', 'a') as f:
              f.write("Saved {} entries in {} and {} in {}".format(len(x), train_path, len(y), test_path))
              f.write("\n")
  
  def tf_record_generation(self,train_json_path,test_json_path,userId,transactionId):
    
    self.images_train_path='/AI_OD/root/lab/prime_team_projects/data/'
    self.images_test_path='/AI_OD/root/lab/prime_team_projects/data/'
    
    self.output_directory= '/home/ob_reserved_cu/scripts/static/store_data/'+userId +'/'+transactionId + '/tf_record/'
    if not os.path.exists(self.output_directory):
      os.makedirs(self.output_directory)
    
  
    train_output_path, testdev_output_path = create_coco_tf_record.create_tf_record_dirs(self.output_directory)
    create_coco_tf_record._create_tf_record_from_coco_annotations(
      annotations_file = train_json_path,
      image_dir=self.images_train_path,
      output_path=train_output_path,
      include_masks=False,
      num_shards=100,
      keypoint_annotations_file='',
      densepose_annotations_file='',
      remove_non_person_annotations=False,
      remove_non_person_images=False)

    create_coco_tf_record._create_tf_record_from_coco_annotations(
      annotations_file=test_json_path, 
      image_dir=self.images_test_path,
      output_path=testdev_output_path,
      include_masks=False,
      num_shards=50)

    return self.output_directory    

  def update_config(self,configFilePath,num_classes,batch_size,learning_rate_base,total_steps,saved_path_pbtxt,tf_record_path,userId,transactionId):
    configs = config_util.get_configs_from_pipeline_file(configFilePath)

    update_config_path='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/updatedconfig'
    updated_path='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/updatedconfig/pipeline.config'
    model_config = configs['model']
    train_config = configs['train_config']
    eval_config=configs['eval_config']
    train_input=configs['train_input_config']
    eval_input=configs['eval_input_configs']

    print(train_input.label_map_path)
  
    configs['model'].faster_rcnn.num_classes=num_classes
    configs['train_config'].batch_size=batch_size
    configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base=learning_rate_base
    configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps=total_steps
    configs['train_config'].num_steps=total_steps
    configs['train_input_config'].label_map_path=saved_path_pbtxt
    print("upated",train_input.label_map_path)
    configs['train_input_config'].tf_record_input_reader.input_path[0]=tf_record_path + 'train.record-?????-of-00100'
    configs['eval_input_configs'][0].label_map_path=saved_path_pbtxt
    print("upated",eval_input[0].label_map_path)
    configs['eval_input_configs'][0].tf_record_input_reader.input_path[0]=tf_record_path + 'test.record-?????-of-00050'
    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, update_config_path)

    return updated_path


    



  def num_cls(self,userId,transactionId):
    json_path='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/class_length.json'
    with open(json_path, "r") as outfile:
      data = json.load(outfile)
      num_classes=data['class_list']
    return num_classes

  def verify_token(self,headers):
        verify_token=headers['token']
        #    ip=headers['ip']
        ip="10.7.246.9" 
        valid_users=requests.post(f"http://{ip}/NslIotHubAPI/authenticateToken?token={verify_token}").json()
        if valid_users['message']=='Token Valid':
            return True
        else:
            return False

  @api.expect(frcnn_resources)
  def post(self):
      # Open json
      try:
          headers=request.headers
          if not self.verify_token(headers):
            return {"message":"Token not valid"}

          data=request.get_json()

          def long_running_task(**kwargs):
            your_params = kwargs.get('post_data', {})
            full_output_dict={
                    'SavedModel': '',
                    'PbtxtPath': ''
              
            }
            configFilePath = "/home/ob_reserved_cu/scripts/models/research/object_detection/configs/tf2/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.config"

            configs = config_util.get_configs_from_pipeline_file(configFilePath)
            
            userId=data['userId']
            transactionId=data['transactionId']
            learning_rate_base=data['Learning_rate']
            total_steps=data['total_steps']
            batch_size=data['batch_size']
            input_dict=data['input_dict']
            input_dict=r'{}'.format(input_dict)
            input_dict=json.loads(input_dict)
            solutionId=input_dict['solutionId']

            for entity in input_dict['changeUnitEntities']:
              if "INPUT" in entity["entityLayer"]:
                  input_entity=entity
              if 'TRIGGERCES' in entity["entityLayer"]:
                  output_entity=entity
                

            self.train_path='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/train.json'
            self.test_path='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+ '/test.json'

            self.model_dir='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/check_point/'
            self.output_inference_dir='/home/ob_reserved_cu/scripts/static/store_data/' + userId + "/"+ transactionId+'/saved_model/'
            
           
            

            self.Check_point_dir=self.model_dir

            if not os.path.exists(self.model_dir):
              os.makedirs(self.model_dir)

            if not os.path.exists(self.output_inference_dir):
              os.makedirs(self.output_inference_dir)
              
            self.destination= self.output_inference_dir + userId + "_" + "Fasterrcnn" + "_" + "custom_classes" + "_" +transactionId + "_trained_model_v1"
            annotation_path=self.checkfile(userId,transactionId)
            #pbtxt file generation
            saved_path_pbtxt=self.generate_pbtxt(annotation_path,userId,transactionId)
            
            print("final_output--------------------------------------------------",full_output_dict)
            if os.path.exists(self.train_path) and os.path.exists(self.test_path):
              print("paths_exists")
                
            else:
              # data split
              self.split_fun(annotation_path,self.train_path,self.test_path) 
              

          #tf_record generation 
            self.tf_record_path=self.tf_record_generation(self.train_path,self.test_path,userId,transactionId)  
            num_classes=self.num_cls(userId,transactionId)
            update_config_path=self.update_config(configFilePath,num_classes,batch_size,learning_rate_base,total_steps,saved_path_pbtxt,self.tf_record_path,userId,transactionId)  
            
            model_main_tf2.train_faster(pipeline_config_path = update_config_path,\
                                        model_dir = self.model_dir,\
                                        num_train_steps = total_steps,\
                                        checkpoint_dir = None)
            
            
            model_main_tf2.train_faster(update_config_path,self.model_dir,total_steps,self.Check_point_dir)
            exporter_main_v2.main(trained_checkpoint_dir=self.model_dir,pipeline_config_path=update_config_path,output_directory=self.output_inference_dir)

            shutil.make_archive(self.destination, 'zip', self.output_inference_dir)

            # full_output_dict['SavedModel']='http://164.52.213.64:5331/static/store_data/' + userId + "/"+ transactionId+'/saved_model/saved_model.zip'
            # full_output_dict['PbtxtPath']='http://164.52.213.64:5331/static/store_data/' + userId + "/"+ transactionId+'/label_map.pbtxt'  
 
            full_output_dict={
             "SavedModel" : 'http://164.52.213.64:5331/static/store_data/' + userId + "/"+ transactionId+ '/saved_model/'+userId+"_"+"Fasterrcnn"+"_"+"custom_classes"+"_"+transactionId+"_trained_model_v1"+'.zip',
             "PbtxtPath":'http://164.52.213.64:5331/static/store_data/' + userId + "/"+ transactionId+'/'+userId+"_"+"Fasterrcnn"+"_"+"custom_classes"+"_"+transactionId+"_trained_model_v1"+'.pbtxt'
              }

            kafka_utils.send_response_from_kafka(input_dict,input_entity,output_entity,userId,solutionId,transactionId,return_dict=full_output_dict,error_message=" ")

            
            return full_output_dict
          

          thread = threading.Thread(target=long_running_task, kwargs={

                        'post_data': data})

          thread.start()

          print("thread")

          return {"Message": "It takes sometime"}
      

      except Exception as e:
          if input_dict['isAsync']:
            kafka_utils.send_response_from_kafka(input_dict,input_entity,output_entity,userId,solutionId,transactionId,error_message=str(e))
          else:
              raise Exception(str(e))

            # full_output_dict= {
            #         "status": 500,
            #         "message": str(e),
            #         "result": []
            # }
            # return full_output_dict


        
                
          # unique_filename = str(uuid.uuid4())
          
      
       



  

if __name__ == '__main__':
  flask_app.run(host='0.0.0.0', port=5331, debug=True, use_reloader=False, threaded=True)                                              