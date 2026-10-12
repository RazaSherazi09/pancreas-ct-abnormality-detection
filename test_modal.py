import modal

# 1. Define your cloud application name
app = modal.App("sanity-check-test")

# 2. Tell Modal what packages the cloud container needs
test_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch"
)

# 3. This function runs on Modal's cloud GPU (Tesla T4)
@app.function(image=test_image, gpu="T4")
def check_cloud_gpu():
    import torch

    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU Detected"
    cuda_version = torch.version.cuda
    
    return {
        "status": "Success",
        "gpu_available": gpu_available,
        "device_name": device_name,
        "cuda_version": cuda_version,
    }

# 4. Entrypoint triggered from your local terminal
@app.local_entrypoint()
def main():
    print("\n[Local] Dispatching job to Modal cloud...")
    result = check_cloud_gpu.remote()
    print("[Local] Response received from Modal cloud:")
    print("--------------------------------------------------")
    print(f" Status:       {result['status']}")
    print(f" GPU Found:    {result['gpu_available']}")
    print(f" Device Name:  {result['device_name']}")
    print(f" CUDA Version: {result['cuda_version']}")
    print("--------------------------------------------------\n")