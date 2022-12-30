import uuid
import time
import json
from kafka import KafkaProducer
bootstrap_server="10.7.246.9:9092"
consumer_group_id="groupId0"
topic="ailabs-notification"

#callback success if the message is published to respective topic
def success(metadata):
    print("published for topic",metadata.topic)

#handling kafka exception
def error(exception):
    print(exception)


def send_response_from_kafka(input_dict,input_entity,output_entity,userId,solutionId,transactionId,return_dict="",error_message=" "):
    if error_message==" ":
        input_dict["changeUnitEntities"]=[]
        input_dict["changeUnitEntities"].append(input_entity)
        for count,attributeout in enumerate(output_entity['attributeList']):
            output_entity['attributeList'][count]['defaultValue']=return_dict["".join(output_entity['attributeList'][count]['attributeName'].split())]    
        input_dict["changeUnitEntities"].append(output_entity)
        status=200
        content=f"Execution of {input_dict['cuName']} is completed"
        changeUnitData=input_dict
    else:
        status=500
        content="ERROR OCCURED:"+error_message
        changeUnitData=input_dict
    #Defining kafka inputs
    isExecutable=True
    kafka_message={}
    myuuid = uuid.uuid4()
    title=f"{input_dict['cuName']}"
    timestamp = int(time.time()*1000.0)
    kafka_message={'id':str(myuuid),"title":title,"content":content,"userId":userId,"transactionId":transactionId,'solutionId':solutionId, "timestamp":timestamp,"changeUnitData":changeUnitData,"isExecutable":isExecutable,'status':status}
    print("Kafka-message",kafka_message)
    #Define kafka producer
    f=open("kafka.json",'w')
    f.write(json.dumps(kafka_message))
    producer = KafkaProducer(bootstrap_servers=bootstrap_server, api_version=(20, 2, 1), value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    producer.send(topic,kafka_message).add_callback(success).add_errback(error)
    print("sent message successfully")