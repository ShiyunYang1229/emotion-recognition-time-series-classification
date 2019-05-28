"""
class Dynamo :
    - create/retrieve table on DynamoDB
    - add item to table
    - query table to retrieve items
"""

import numpy as np
import time,json,sys

import boto3
from boto3.dynamodb.conditions import Key,Attr
import get_aws as aws

DYNAMO_TABLE_NAME = "mindWave"

class Dynamo:
    def __init__(self):
        dynamo = aws.getResource('dynamodb')

        # First try and create the table
        try:
            table = dynamo.create_table(
                TableName=DYNAMO_TABLE_NAME,
                KeySchema=[
                    {
                        'AttributeName': 'userName',
                        'KeyType': 'HASH'  # Partition Key
                    },
                    {
                        'AttributeName' : 'time',
                        'KeyType' : 'RANGE'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'userName',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'time',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 10,
                    'WriteCapacityUnits': 10
                }
            )
            while table.table_status != 'ACTIVE':
                table.reload()
            print ('Table ' + DYNAMO_TABLE_NAME + ' has been created.')
        except Exception as e:
            # print e
            table = dynamo.Table(DYNAMO_TABLE_NAME)
            print ('Table ' + DYNAMO_TABLE_NAME + ' has been retrieved.')
        self.table_dynamo = table

    def dynamoAdd(self, userName, time, data, result):
        self.table_dynamo.put_item(
            Item = {
               'userName' : userName,
               'time' : time,
               'data' : data,
               'result': result
            }
        )

    def dynamoQuery(self, userName):
        response = self.table_dynamo.query(
            KeyConditionExpression = Key('userName').eq(userName)
        )
        items = response['Items']
        return items