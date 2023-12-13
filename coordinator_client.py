import grpc
import asyncio
import helloworld_pb2  # 导入你的 gRPC 自动生成的协议文件
import helloworld_pb2_grpc # 导入你的 gRPC 自动生成的服务文件

async def make_request(server_address, request_data):
    # 创建 gRPC 通道
    channel = grpc.aio.insecure_channel(server_address)

    # 创建 gRPC 客户端
    stub = helloworld_pb2_grpc.GreeterStub(channel)

    # 创建请求对象
    request = helloworld_pb2.HelloRequest(name=request_data)

    # 使用 await 关键字调用异步 gRPC 方法
    response = await stub.SayHello(request)

    # 处理响应
    print(f"Received response from {server_address}: {response.message}")

    # 关闭通道
    await channel.close()

async def main():
    # 定义多个服务器地址
    server_addresses = ["127.0.0.1:50051", "127.0.0.1:50052", "127.0.0.1:50053"]

    # 定义要发送的请求数据
    request_datas = ["Hello gRPC from 50051!", "Hello gRPC from 50052!", "Hello gRPC from 50053!"]

    # 创建任务列表
    tasks = [make_request(address, request_data) for address, request_data in zip(server_addresses, request_datas)]

    # 使用 asyncio.gather 同时运行多个任务
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
