import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_file_path, engine_file_path, max_batch_size, max_workspace_size):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Error: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        config.set_flag(trt.BuilderFlag.FP16)
        # engine = builder.build_engine(network, config)
        engine = builder.build_serialized_network(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine, batch_size):
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = [], [], [], [], []

    for binding in engine:
        shape = (batch_size,)+tuple(engine.get_tensor_shape(binding))
        size = trt.volume(shape)
        dtype = engine.get_tensor_dtype(binding)
        host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings

def inference(context, host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, stream, input_frame):
    input_frame = cv2.resize(input_frame, (416, 416))
    input_frame = input_frame.transpose((2, 0, 1))
    input_frame = input_frame.astype(np.float32)
    input_frame /= 255.0
    print(f"input_frame: {input_frame.shape}")
    np.copyto(host_inputs[0], input_frame.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    print(host_outputs[0])
    return host_outputs[0]

if __name__ == '__main__':
    engine_file = "model_fp16_large_workspace.engine"
    video_path = "2024-05-27 07-57-00.mp4"
    fps_results = "fps_results.txt"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video. ({video_path})")
        exit()
    
    print(f"Loading engine: {engine_file}")
    engine = load_engine(engine_file)
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = allocate_buffers(engine, 128)
    
    with open(fps_results, "w") as f:
        with engine.create_execution_context() as context:
            while True:
                total_frames = 0
                start_time = time.time()
                
                while time.time() < start_time + 30:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print(f"shape: {frame.shape}")
                    output = inference(context, host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, cuda.Stream(), input_frame=frame)
                    total_frames += 1
                
                elapsed_time = time.time() - start_time
                fps = total_frames / elapsed_time
                print(f"FPS: {fps:.2f}")
                f.write(f"FPS: {fps:.2f}\n")
                f.flush()
    cap.release()