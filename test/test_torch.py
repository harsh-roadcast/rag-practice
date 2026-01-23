import torch
if torch.backends.mps.is_available():
    print("✅ Success! Using Apple M2 GPU (MPS).")
else:
    print("❌ Warning: Running on CPU. Slower, but still faster than the Dell.")