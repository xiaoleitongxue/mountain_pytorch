# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import objectdetection_pb2 as objectdetection__pb2


class InferenceStub(object):
    """The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.inference = channel.unary_unary(
                '/objectdetection.Inference/inference',
                request_serializer=objectdetection__pb2.InferenceRequest.SerializeToString,
                response_deserializer=objectdetection__pb2.InferenceReply.FromString,
                )
        self.inferenceStreamReply = channel.unary_stream(
                '/objectdetection.Inference/inferenceStreamReply',
                request_serializer=objectdetection__pb2.InferenceRequest.SerializeToString,
                response_deserializer=objectdetection__pb2.InferenceReply.FromString,
                )


class InferenceServicer(object):
    """The greeting service definition.
    """

    def inference(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def inferenceStreamReply(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'inference': grpc.unary_unary_rpc_method_handler(
                    servicer.inference,
                    request_deserializer=objectdetection__pb2.InferenceRequest.FromString,
                    response_serializer=objectdetection__pb2.InferenceReply.SerializeToString,
            ),
            'inferenceStreamReply': grpc.unary_stream_rpc_method_handler(
                    servicer.inferenceStreamReply,
                    request_deserializer=objectdetection__pb2.InferenceRequest.FromString,
                    response_serializer=objectdetection__pb2.InferenceReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'objectdetection.Inference', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Inference(object):
    """The greeting service definition.
    """

    @staticmethod
    def inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/objectdetection.Inference/inference',
            objectdetection__pb2.InferenceRequest.SerializeToString,
            objectdetection__pb2.InferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def inferenceStreamReply(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/objectdetection.Inference/inferenceStreamReply',
            objectdetection__pb2.InferenceRequest.SerializeToString,
            objectdetection__pb2.InferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)