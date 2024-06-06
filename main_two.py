import os
import time
import ffmpeg
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
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
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
    np.copyto(host_inputs[0], input_frame.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    return host_outputs[0]

def process_video(input_file, engine_file, fps_results):
    # Open the video file
    video = ffmpeg.input(input_file)

    # Set up the video processing pipeline
    pipeline = video.output('pipe:', format='rawvideo', pix_fmt='rgb24')

    # Load the TensorRT engine
    engine = load_engine(engine_file)
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = allocate_buffers(engine, 128)

    with open(fps_results, "w") as f:
        with engine.create_execution_context() as context:
            while True:
                total_frames = 0
                start_time = time.time()

                # Run the FFmpeg command and capture the output
                out, _ = pipeline.run(capture_stdout=True)

                while time.time() < start_time + 30:
                    # Extract a single frame
                    frame_size = 416 * 416 * 3
                    if len(out) < frame_size:
                        break
                    frame_data = out[:frame_size]
                    out = out[frame_size:]

                    # Process the frame data
                    frame = np.frombuffer(frame_data, np.uint8).reshape(416, 416, 3)
                    input_frame = frame.astype(np.float32)

                    # Run inference
                    output = inference(context, host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, cuda.Stream(), input_frame)
                    total_frames += 1

                elapsed_time = time.time() - start_time
                fps = total_frames / elapsed_time
                print(f"FPS: {fps:.2f}")
                f.write(f"FPS: {fps:.2f}\n")
                f.flush()  # Flush the buffer to write immediately

                if len(out) == 0:
                    break

    # Release resources
    cuda.Context.pop()

if __name__ == '__main__':
    engine_file = "model_fp16_large_workspace.engine"
    video_path = "2024-05-27 07-57-00.mp4"
    fps_results = "fps_results.txt"

    process_video(video_path, engine_file, fps_results)
