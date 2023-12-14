from urllib import request
import grpc
import asyncio
import helloworld_pb2
import objectdetection_pb2  # 导入你的 gRPC 自动生成的协议文件
import objectdetection_pb2_grpc
import helloworld_pb2_grpc  # 导入你的 gRPC 自动生成的服务文件
import torch

import utils


async def make_request(server_address, name: str, data: bytes) -> torch.Tensor:
    # 创建 gRPC 通道
    options = [('grpc.max_send_message_length', 4309960),
               ('grpc.max_receive_message_length', 4309960)]
    channel = grpc.aio.insecure_channel(target=server_address, options=options)

    # 创建 gRPC 客户端
    # stub = helloworld_pb2_grpc.GreeterStub(channel)
    stub = objectdetection_pb2_grpc.InferenceStub(channel)
    # request = helloworld_pb2.HelloRequest(name=name, file=tensor)
    request = objectdetection_pb2.InferenceRequest(name=name, data=data)
    # 使用 await 关键字调用异步 gRPC 方法
    # response = await stub.SayHello(request)
    response = await stub.inference(request)
    # 处理响应
    print(f"Received response from {server_address}: {response.name}")

    # 关闭通道
    await channel.close()

    return utils.bytes2tensor(response.data)


async def main():
    # 定义多个服务器地址
    server_addresses = ["127.0.0.1:50051", "127.0.0.1:50052",
                        "127.0.0.1:50053", "127.0.0.1:50054"]
    # 定义要发送的请求数据
    image_list = utils.slice_input('./images/dog.jpg')
    bytesed_image_list = list(map(utils.tensor2bytes, image_list))
    # bytesed_image_list = [bytes(3), bytes(3), bytes(3), bytes(3)]
    request_names = ["Hello gRPC from 50051!",
                     "Hello gRPC from 50052!", "Hello gRPC from 50053!", "Hello gRPC from 50054!"]

    # 创建任务列表
    tasks = [make_request(address, name, data) for address,
             name, data in zip(server_addresses, request_names, bytesed_image_list)]

    # # 使用 asyncio.gather 同时运行多个任务
    await asyncio.gather(*tasks)

    for result in tasks:
        print(result)

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
