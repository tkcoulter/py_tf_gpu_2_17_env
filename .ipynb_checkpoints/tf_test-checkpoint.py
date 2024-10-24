import tensorflow as tf

print(tf.__version__)

print("\nGPU Information:")
print("-" * 50)
    
# Check if TensorFlow can see any GPUs
physical_devices = tf.config.list_physical_devices()
print("All physical devices:", physical_devices)
    
physical_gpus = tf.config.list_physical_devices('GPU')
print("\nNumber of GPUs available:", len(physical_gpus))
    
if len(physical_gpus) > 0:
    print("\nGPU Devices:")
    for gpu in physical_gpus:
        print(f"  {gpu}")
            
    # Try to get GPU device details
    try:
        for gpu in physical_gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"\nGPU Details: {details}")
    except:
        print("Could not get detailed GPU information")
    
    # Check if TensorFlow is using GPU
    print("\nTensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("TensorFlow GPU available:", tf.test.is_gpu_available())
    
# Print device placement
print("\nDevice Placement:")
with tf.device('/CPU:0'):
    print("CPU operation:", tf.random.normal([1]).device)
try:
    with tf.device('/GPU:0'):
        print("GPU operation:", tf.random.normal([1]).device)
except:
    print("GPU operation: Not available")