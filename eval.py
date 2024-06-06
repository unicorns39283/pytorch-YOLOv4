import subprocess
import sys
import time, cv2

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy"])
    print("numpy installed successfully")
    import numpy as np
    
try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
    print("onnxruntime installed successfully")
    import onnxruntime as ort

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError:
    print("Installing pycuda...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pycuda"])
    print("pycuda installed successfully")
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

from typing import List

def nms_cpu(boxes: np.ndarray, confs: np.ndarray, nms_thresh: float = 0.5, min_mode: bool = False) -> np.ndarray:
    x1: np.ndarray = boxes[:, 0]
    y1: np.ndarray = boxes[:, 1]
    x2: np.ndarray = boxes[:, 2]
    y2: np.ndarray = boxes[:, 3]
    areas: np.ndarray = (x2 - x1) * (y2 - y1)
    order: np.ndarray = confs.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        idx_self: int = order[0]
        idx_other: np.ndarray = order[1:]
        keep.append(idx_self)
        xx1: np.ndarray = np.maximum(x1[idx_self], x1[idx_other])
        yy1: np.ndarray = np.maximum(y1[idx_self], y1[idx_other])
        xx2: np.ndarray = np.minimum(x2[idx_self], x2[idx_other])
        yy2: np.ndarray = np.minimum(y2[idx_self], y2[idx_other])
        w: np.ndarray = np.maximum(0.0, xx2 - xx1)
        h: np.ndarray = np.maximum(0.0, yy2 - yy1)
        inter: np.ndarray = w * h
        if min_mode: over: np.ndarray = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else: over: np.ndarray = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds: np.ndarray = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)
def nms_gpu(boxes: np.ndarray, confs: np.ndarray, nms_thresh: float = 0.5, min_mode: bool = False) -> np.ndarray:
    if boxes.shape[0] == 0: return np.zeros(0, dtype=np.int32)
    # CUDA kernel for calculating IoU
    iou_kernel = SourceModule("""
    __device__ float iou(float *boxes, int i, int j, int min_mode) {
        __shared__ float s_boxes[256];
        int tx = threadIdx.x, idx = i * 4;
        s_boxes[tx * 4 + 0] = boxes[idx + 0]; s_boxes[tx * 4 + 1] = boxes[idx + 1];
        s_boxes[tx * 4 + 2] = boxes[idx + 2]; s_boxes[tx * 4 + 3] = boxes[idx + 3];
        __syncthreads();
        float xx1 = fmaxf(s_boxes[tx * 4 + 0], boxes[j * 4 + 0]);
        float yy1 = fmaxf(s_boxes[tx * 4 + 1], boxes[j * 4 + 1]);
        float xx2 = fminf(s_boxes[tx * 4 + 2], boxes[j * 4 + 2]);
        float yy2 = fminf(s_boxes[tx * 4 + 3], boxes[j * 4 + 3]);
        float w = fmaxf(0.0f, xx2 - xx1), h = fmaxf(0.0f, yy2 - yy1), inter = w * h;
        float area_i = (s_boxes[tx * 4 + 2] - s_boxes[tx * 4 + 0]) * (s_boxes[tx * 4 + 3] - s_boxes[tx * 4 + 1]);
        float area_j = (boxes[j * 4 + 2] - boxes[j * 4 + 0]) * (boxes[j * 4 + 3] - boxes[j * 4 + 1]);
        return min_mode ? inter / fminf(area_i, area_j) : inter / (area_i + area_j - inter);
    }
    
    __global__ void nms_kernel(float *boxes, float *confs, int *order, int *keep, int num_boxes, float nms_thresh, int min_mode) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_boxes) return;
        int idx_self = order[idx];
        if (idx_self == -1) return;
        for (int i = idx + 1; i < num_boxes; i++) {
            int idx_other = order[i];
            if (idx_other == -1) continue;
            if (iou(boxes, idx_self, idx_other, min_mode) > nms_thresh) atomicExch(&order[i], -1);
        }
    }
    __global__ void count_valid_elements(int *keep, int *count, int num_boxes) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_boxes && keep[idx] != -1) atomicAdd(count, 1);
    }
    """)
    boxes_gpu: cuda.DeviceAllocation = cuda.to_device(boxes.astype(np.float32))
    confs_gpu: cuda.DeviceAllocation = cuda.to_device(confs.astype(np.float32))
    order_gpu: cuda.DeviceAllocation = cuda.to_device(np.argsort(confs)[::-1].astype(np.int32))
    keep_gpu: cuda.DeviceAllocation = cuda.mem_alloc(boxes.shape[0] * 4)  # 4 bytes for int32
    num_boxes: int = boxes.shape[0]
    nms_thresh: np.float32 = np.float32(nms_thresh)
    min_mode_int: np.int32 = np.int32(min_mode)
    block_size: int = 256
    grid_size: int = (num_boxes + block_size - 1) // block_size
    nms_kernel = iou_kernel.get_function("nms_kernel")
    nms_kernel(boxes_gpu, confs_gpu, order_gpu, keep_gpu, np.int32(num_boxes), nms_thresh, min_mode_int, block=(block_size, 1, 1), grid=(grid_size, 1))
    count_gpu: cuda.DeviceAllocation = cuda.mem_alloc(4)  # 4 bytes for int
    cuda.memset_d32(count_gpu, 0, 1)
    count_kernel = iou_kernel.get_function("count_valid_elements")
    count_kernel(keep_gpu, count_gpu, np.int32(num_boxes), block=(block_size, 1, 1), grid=(grid_size, 1))
    count: np.ndarray = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(count, count_gpu)
    keep: np.ndarray = np.zeros(count[0], dtype=np.int32)
    cuda.memcpy_dtoh(keep, keep_gpu)
    return keep
def post_processing(img: np.ndarray, conf_thres: float, nms_thres: float, output: List[np.ndarray], provider: str) -> List[List[List[float]]]:
    box_array: np.ndarray = output[0]
    confs: np.ndarray = output[1]
    t1: float = time.time()
    if type(box_array).__name__ != 'ndarray':
        box_array, confs = box_array.cpu().detach().numpy(), confs.cpu().detach().numpy()
    num_classes: int = confs.shape[2]
    box_array: np.ndarray = box_array[:, :, 0]
    max_conf: np.ndarray = np.max(confs, axis=2)
    max_id: np.ndarray = np.argmax(confs, axis=2)
    t2: float = time.time()
    bboxes_batch: List[List[List[float]]] = []
    for i in range(box_array.shape[0]):
        argwhere: np.ndarray = max_conf[i] > conf_thres
        l_box_array: np.ndarray = box_array[i, argwhere, :]
        l_max_conf: np.ndarray = max_conf[i, argwhere]
        l_max_id: np.ndarray = max_id[i, argwhere]
        bboxes: List[List[float]] = []
        for j in range(num_classes):
            argwhere: np.ndarray = l_max_id == j
            ll_box_array: np.ndarray = l_box_array[argwhere, :]
            ll_max_conf: np.ndarray = l_max_conf[argwhere]
            ll_max_id: np.ndarray = l_max_id[argwhere]
            keep: np.ndarray = nms_gpu(ll_box_array, ll_max_conf, nms_thres) if provider == 'CUDAExecutionProvider' else nms_cpu(ll_box_array, ll_max_conf, nms_thres)
            if keep.size > 0:
                ll_box_array: np.ndarray = ll_box_array[keep, :]
                ll_max_conf: np.ndarray = ll_max_conf[keep]
                ll_max_id: np.ndarray = ll_max_id[keep]
                bboxes.extend([[ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]] for k in range(ll_box_array.shape[0])])
        bboxes_batch.append(bboxes)
    t3: float = time.time()
    print(f'-----------------------------------\n       max and argmax : {t2 - t1:.6f}\n                  nms : {t3 - t2:.6f}\nPost processing total : {t3 - t1:.6f}\n-----------------------------------')
    return bboxes_batch

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
pp_provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in providers else 'CPUExecutionProvider'
print(f"Using {providers[0]}")
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 8
sess_opts.execution_order = ort.ExecutionOrder.PRIORITY_BASED
sess_opts.enable_profiling = True
session = ort.InferenceSession('yolov4_optimized.onnx', providers=providers, sess_options=sess_opts)
prev_frame_time = 0
new_frame_time = 0
video_path = "2024-05-27 07-57-00.mp4"
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, screenshot = cap.read()
    if not ret: break
    input_data = np.expand_dims(np.transpose(cv2.resize(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB), (416, 416)), (2, 0, 1)), axis=0).astype(np.float32) / 255.0
    outputs = session.run(None, {'input': input_data})
    boxes = post_processing(input_data, 0.5, 0.6, outputs, pp_provider)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")
profile_file = session.end_profiling()
print(f"Profile file: {profile_file}")
cv2.destroyAllWindows()