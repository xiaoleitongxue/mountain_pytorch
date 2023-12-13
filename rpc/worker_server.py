# Copyright 2015 gRPC authors.
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
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
from os import read

import grpc
from torch import tensor
import torch
import grpcInterface as gi

import sys

from model import Test_model
import utils



class Greeter(gi.helloworld_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return gi.helloworld_pb2.HelloReply(message="Hello, %s!" % request.name)

class Inference(gi.objectdetection_pb2_grpc.InferenceServicer):
    def __init__(self) -> None:
        super().__init__()
        self.model = Test_model.Model();
    
    def inference(self, request, context):
        # convert request.tensor to filestream
        tensored_bytes = utils.bytes2tensor(request.data)
        box = self.model.inference(tensored_bytes)
        reply = gi.objectdetection_pb2.InferenceReply(name='Completed inference', data=utils.tensor2bytes(box))
        return reply

def serve(p):
    port = p
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gi.helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    port = sys.argv[1]
    serve(port)