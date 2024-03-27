import json
import multiprocessing
import subprocess
import os
import numpy as np
import random
from copy import deepcopy
import traceback


def create_validation_json(json_train, num_train_samples, num_val_samples):
    """
    Create json for validation. Use train views and interpolate new views.
    """
    json_dict = deepcopy(json_train)
    frames = json_train["frames"]
    json_dict["frames"] = []

    num_train_samples = min(num_train_samples, len(frames))
    train_samples = random.sample(range(len(frames)), num_train_samples)

    for i in train_samples:
        json_dict["frames"].append(frames[i])

    ext = frames[0]["file_path"].split(".")[-1]
    for i in range(num_val_samples):
        views = np.random.choice(range(len(frames)), 2, replace=False)
        interpolated = np.eye(4)

        xforms1 = np.array(frames[views[0]]["transform_matrix"])
        xforms2 = np.array(frames[views[1]]["transform_matrix"])

        interpolated[:3, :3] = xforms1[:3, :3]
        interpolated[:3, 3] = (xforms2[:3, 3] + xforms1[:3, 3]) * 0.5

        json_dict["frames"].append(
            {"file_path": f"val_{i}.{ext}", "transform_matrix": interpolated.tolist()}
        )

    return json_dict


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    data_path = (
        "/home/ubuntu/zubair/NeRF_MAE/data/scannet/scannet_nerf_output_all/scannet_nerf"
    )
    out_dir = "/home/ubuntu/zubair/NeRF_MAE/data/scannet/scannet_rpn_out_all"
    ckpt_dir = "/home/ubuntu/zubair/NeRF_MAE/data/scannet/out_checkpoints"
    bbox_dir = "/data/zubair/scannet_boxes/"

    while True:
        scene = queue.get()
        if scene is None:
            break

        out_name = scene + ".npz"
        if out_name in os.listdir(out_dir):
            print("Scene {} already trained, skipping...".format(scene))
            queue.task_done()
            continue
        else:
            print("Path does not exist for scene:", scene, "training scene...")
        print(scene, gpu)

        print("Training scene....", scene)

        bbox_path = os.path.join(bbox_dir, scene + ".json")
        # Specify the command string with CUDA_VISIBLE_DEVICES and the arguments
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} python ./run_nerf.py extract"
            f" --scene_id {scene} --expname {scene} --data_dir {data_path} --max_res 160 --ckpt_dir {ckpt_dir} --extract_dir {out_dir} --bbox_json {bbox_path}"
        )

        try:
            # subprocess.run(arg_list)
            subprocess.run(command, shell=True)
        except:
            print("Failed to train scene: {}".format(scene))
            traceback.print_exc()

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # data_path = "/wild6d_data/zubair/FRONT3D_render"
    data_path = (
        "/home/ubuntu/zubair/NeRF_MAE/data/scannet/scannet_nerf_output_all/scannet_nerf"
    )
    scenes = os.listdir(data_path)
    scenes = [s for s in scenes if os.path.isdir(os.path.join(data_path, s))]
    scenes = sorted(scenes)

    # scenes = scenes[:14]

    worker_per_gpu = 1

    gpus_available = [0, 2, 3, 4, 5]
    num_gpus = len(gpus_available)
    gpu_start = gpus_available[0]

    workers = num_gpus * worker_per_gpu
    # Start worker processes on each of the GPUs
    for gpu_i in range(num_gpus):
        gpu_id = gpus_available[gpu_i]
        for worker_i in range(worker_per_gpu):
            worker_i = (gpu_i - gpu_start) * worker_per_gpu + worker_i
            # worker_id = (gpu_i - gpu_start) * worker_per_gpu + worker_i
            # worker_i = gpu_i * worker_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_id)
            )
            process.daemon = True
            process.start()

    # Add items to the queue

    for item in scenes:
        queue.put(item)

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(num_gpus * worker_per_gpu):
        queue.put(None)
