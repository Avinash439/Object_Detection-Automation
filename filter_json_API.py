from flask import Flask, jsonify, make_response, request
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import BadRequest
import pandas as pd
import time
import os
import json
import threading
from pathlib import Path
import uuid
import numpy as np
import matplotlib.pyplot as plt
import traceback
import random
import cv2
import requests
import kafka_utils 

flask_app = Flask(__name__,static_url_path='/static')

api = Api(app = flask_app, version = '1.0', title = 'Filter_Json', description = 'Localizing objects in an image')
Json_resources = api.model('GridCountResources',
                                          {
                                            
                                            'userId': fields.String(required = True, default = ' ', description="Input userId", help="input_path"),
                                            'transactionId': fields.String(required = True, default = '', description = 'transactionid'),
                                            'input_class_list':fields.List(fields.String,description="select the required classes to filter"),
                                            'input_dict':fields.String(required=True, default='',description="kafka_input")
                                          })
                                              

@api.route('/Class_info')
class Object_Count(Resource):

    
    def _process_info(self):
        self.info = self.coco['info']
        
    def _process_licenses(self):
        self.licenses = self.coco['licenses']
    
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category['name'])
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
                
    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        missing_categories = set(self.filter_categories) - self.category_set
        if len(missing_categories) > 0:

            return 0   
            
            # resp = make_response(f'Did not find categories: {missing_categories}')
            # resp.headers['Access-Control-Allow-Origin'] = '*'
            # return resp
            # print(f'Did not find categories: {missing_categories}')
            # should_continue = input('Continue? (y/n) ').lower()
            # if should_continue != 'y' and should_continue != 'yes':
            #     print('Quitting early.')
            #     quit()

        self.new_category_map = dict()
        new_id = 0
        for key, item in self.categories.items():
            if item['name'] in self.filter_categories:
                self.new_category_map[key] = new_id
                new_id += 1

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = new_id
            self.new_categories.append(new_category)
        return 1    

    def _filter_annotations(self):
        """ Create new collection of annotations matching category ids
            Keep track of image ids matching annotations
        """
        self.new_segmentations = []
        self.new_image_ids = set()
        ann_count = 0
        for image_id, segmentation_list in self.segmentations.items():
            for segmentation in segmentation_list:
                original_seg_cat = segmentation['category_id']
                if original_seg_cat in self.new_category_map.keys():
                    new_segmentation = dict(segmentation)
                    new_segmentation['category_id'] = self.new_category_map[original_seg_cat]
                    self.new_segmentations.append(new_segmentation)
                    self.new_image_ids.add(image_id)
                    ann_count += 1
        print('Total annotations are - ', ann_count)

    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        img_count = 0
        for image_id in self.new_image_ids:
            self.new_images.append(self.images[image_id])
            img_count += 1
        print('Total images are - ', img_count)
        
    def UploadFile(self,userId,transactionId):
        path='/home/ob_reserved_cu/scripts/static/store_data/'+userId
        
        # Check if dierctory exists, if not then create new dirctory
        if not os.path.exists(path):
            os.makedirs(path)
        path='/home/ob_reserved_cu/scripts/static/store_data/'+userId+"/"+transactionId
        if not os.path.exists(path):
            os.makedirs(path)

        url_path="static/store_data/"+userId+"/"+transactionId
        return path, url_path
    
    def stats(self,path,plot_path,url_path):
        
        with open(path, "r") as outfile:
            data = json.load(outfile)

        unique_filename = str(uuid.uuid4())
        save_path=plot_path+ "/stats.pdf"
        plot_url_path=url_path+ "/stats.pdf"
        n_images = len(data['images'])
        n_boxes = len(data['annotations'])
        n_categ = len(data['categories'])


        Dict=data['annotations']


        df= pd.DataFrame(Dict)
        images_list=[]

        bbox=df.groupby("category_id")
        bbox_list=[len(bbox.get_group(x)) for x in bbox.groups]
        categ_map = {x['id']: x['name'] for x in data['categories']}

        Bb_report=dict(zip(categ_map.values(),bbox_list))
        print(Bb_report)


        label_id_num_img_dict = {i:[] for i in range(n_categ)}
        for row, col in df.iterrows():
            if col['image_id'] not in label_id_num_img_dict[col['category_id']]:
                label_id_num_img_dict[col['category_id']].append(col['image_id'])



        for label, li in label_id_num_img_dict.items():
            images_list.append(len(li))
            


        image_report=dict(zip(categ_map.values(),images_list))
        print(image_report)


        X = categ_map.values()
        imgs= images_list
        no_of_bbox=bbox_list
        
        X_axis = np.arange(len(X))
        
        plt.bar(X_axis - 0.2, imgs, 0.4, label = 'no_of_images')
        plt.bar(X_axis + 0.2, no_of_bbox, 0.4, label = 'no_of_bounding_boxes')
        plt.xticks(X_axis, X)
        plt.xlabel("classes")
        plt.ylabel("Number of images")
        plt.title("Number of bounding boxes in each class")
        plt.legend()
        
        plt.savefig(save_path)
        plt.close()
        return plot_url_path


    def test_annotation(self,path,n,urls_path,url_path):
        with open(path, "r") as outfile:
            data = json.load(outfile)


        n_images = len(data['images'])
        n_boxes = len(data['annotations'])
        n_categ = len(data['categories'])


        Dict=data['annotations']
        Dict2=data['images']


        anno_df= pd.DataFrame(Dict)
        images_df=pd.DataFrame(Dict2)

        # print('Annotation df')
        # print(anno_df.head(n=10))
        # print(' ')
        # print('Image Df')
        # print(images_df.head())


        image_id_groups=anno_df.groupby("image_id")
        uniq_img_ids = anno_df['image_id'].unique()
        

        image_name_bboxes_list_dict = {}
        for index, uniq_img_id in enumerate(uniq_img_ids):
            curr_img_id_group = image_id_groups.get_group(uniq_img_id)
            image_name_bboxes_list_dict[images_df.loc[images_df['id'] == uniq_img_id]['file_name'].values.tolist()[0]] = curr_img_id_group['bbox'].values.tolist()
            if len(image_name_bboxes_list_dict)>5000:
                break

        path='/home/od_full_data/data/'
        url=[]
        for i in range(n):
            image, bounding_box = random.choice(list(image_name_bboxes_list_dict.items()))
            print(image)
            print(bounding_box)
            img=cv2.imread(path+image)
            size=img.shape
            image_width=size[0]
            image_height=size[1]
            print(size)
            for box in bounding_box:
                start = (int(box[0]),int(box[1]))
                end = (int(box[2])+int(box[0]),int(box[3])+int(box[1]))
                cv2.rectangle(img,start,end,(0,0,255),7)

            unique_filename = str(uuid.uuid4())
            file_path=urls_path+"/"+unique_filename+".png"

            url.append('http://164.52.213.64:5334/'+url_path+"/"+unique_filename+".png")
            print(url)
            cv2.imwrite(file_path,img)
        return url    

    def verify_token(self,headers):
        verify_token=headers['token']
        #    ip=headers['ip']
        ip="10.7.246.9" 
        valid_users=requests.post(f"http://{ip}/NslIotHubAPI/authenticateToken?token={verify_token}").json()
        if valid_users['message']=='Token Valid':
            return True
        else:
            return False

        
    @api.expect(Json_resources)
    def post(self):
        # Open json
        try:
            headers=request.headers
            if not self.verify_token(headers):
                return {"message":"Token not valid"}
            data=request.get_json()

            def long_running_task(**kwargs):
                your_params = kwargs.get('post_data', {})

                userId=data['userId']
                transactionId=data['transactionId']
                input_dict=data['input_dict']
                input_dict=r'{}'.format(input_dict)
                input_dict=json.loads(input_dict)
                solutionId=input_dict['solutionId']

                for entity in input_dict['changeUnitEntities']:
                    if "INPUT" in entity["entityLayer"]:
                        input_entity=entity
                    if 'TRIGGERCES' in entity["entityLayer"]:
                        output_entity=entity
                
            

                path,url_path=self.UploadFile(userId,transactionId)
                unique_filename = str(uuid.uuid4())
                self.input_json_path = "/home/ob_reserved_cu/jsons/project9_i1_merged3.json"
                self.output_json_path = path +"/trainable.json"
                self.filter_categories = data['input_class_list']
                json_path='/home/ob_reserved_cu/scripts/static/store_data/'+userId +'/' +transactionId+ '/class_length.json'

                len_class_list={'class_list':''}    

              
                len_class_list['class_list']=len(self.filter_categories)
                with open(json_path, 'w') as outfile:
                    json.dump(len_class_list, outfile)

                print("paths", self.input_json_path)
                print("paths", self.output_json_path)
                print("paths", self.filter_categories)


                print(type(self.filter_categories))
                # with open(args.categories) as f:
                #     object_names = f.read().splitlines()
                    
                object_names=self.filter_categories 

                # Verify input path exists
                if not os.path.exists(self.input_json_path):
                    print('Input json path not found.')
                    print('Quitting early.')
                    quit()

                # Verify output path does not already exist
                # if os.path.exists(self.output_json_path):
                #     # should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
                #     if should_continue != 'y' and should_continue != 'yes':
                #         print('Quitting early.')
                #         quit()
                
                # Load the json
                print('Loading json file...')
                with open(self.input_json_path) as json_file:
                    self.coco = json.load(json_file)
                
                # Process the json
                print('Processing input json...')
                self._process_info()
                self._process_licenses()
                self._process_categories()
                self._process_images()
                self._process_segmentations()

                # Filter to specific categories
                print('Filtering...')
                category_rt=self._filter_categories()
                if category_rt == 1:
                    self._filter_annotations()
                    self._filter_images()

                    # Build new JSON
                    new_master_json = {
                        'info': self.info,
                        'licenses': self.licenses,
                        'images': self.new_images,
                        'annotations': self.new_segmentations,
                        'categories': self.new_categories
                    }
                    

                    # Write the JSON to a file
                    print('Saving new json file...')
                    with open(self.output_json_path, 'w+') as output_file:
                        json.dump(new_master_json, output_file)

                    print('Filtered json saved.')
                    output={"plot_path":'',"visualization":[]}
                    plot=self.stats(self.output_json_path,path,url_path)
                    urls=self.test_annotation(self.output_json_path,len(self.filter_categories),path,url_path)
                    print(plot)
                    output['PlotPath']='http://164.52.213.64:5334/'+ plot
                    output['Visualization']= "," .join(urls)

                    output_dict= output
                else:
                    output_dict= jsonify({'error':'There are missing categories'})
                    
                    
                # full_output_dict= {
                #         "status": 200,
                #         "message": "Reserved CU Executed Successfully",
                #         "result": [output_dict]
                # }
                


                kafka_utils.send_response_from_kafka(input_dict,input_entity,output_entity,userId,solutionId,transactionId,return_dict=output,error_message=" ")
                # print(full_output_dict)
                # return full_output_dict

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




        # except Exception as e:
        #     print(traceback.format_exc())
        #     resp = make_response('Error is --> ', e)
        #     resp.headers['Access-Control-Allow-Origin'] = '*'
        #     return resp




  

if __name__ == '__main__':

    flask_app.run(host='0.0.0.0', port=5334, debug=True, use_reloader=False, threaded=True)                                              