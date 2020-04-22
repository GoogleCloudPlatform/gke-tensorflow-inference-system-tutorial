# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import grpc
import numpy
import os
import time

import tensorrtserver.api.model_config_pb2 as model_config
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc

from PIL import Image

DATA_DIR = 'data'
DATA_SIZE = 3
TIMEOUT_MSEC = 2000
MODEL_NAME = os.environ['MODEL_NAME']
SERVER_ADDRESS = os.environ['SERVER_ADDRESS']

def init_stub(server_address):
  return stub

def init_request(model_name, data_dir, data_size):
  image_id = numpy.random.randint(data_size) + 1
  image_file = '{}/{:0>5}.jpg'.format(data_dir, image_id)
  image = Image.open(image_file).convert('RGB')
  image = image.resize((224, 224), Image.BILINEAR)
  image = numpy.array(image).astype(numpy.float32)

  input_bytes = image.tobytes()
  request = grpc_service_pb2.InferRequest()

  request.model_name = model_name
  request.model_version = -1
  request.meta_data.batch_size = 1

  output_message = api_pb2.InferRequestHeader.Output()
  output_message.name = 'probabilities'
  output_message.cls.count = 1

  request.meta_data.output.extend([output_message])
  request.meta_data.input.add(name='input')
  request.raw_input.extend([input_bytes])

  return request

def main():
  hostport = "{}:8001".format(SERVER_ADDRESS)
  channel = grpc.insecure_channel(hostport)
  stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
  request = init_request(MODEL_NAME, DATA_DIR, DATA_SIZE)

  while(True):
    request_per_sec = int(os.environ['REQUEST_PER_SEC'])
    result = stub.Infer(request, TIMEOUT_MSEC)
    time.sleep(1./request_per_sec)

if __name__ == "__main__":
  main()
