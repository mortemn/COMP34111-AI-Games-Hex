import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import time
from agents.Group25.azalea_network import load_azalea_model

print("Loading model...")
model_path = "agents/Group25/models/hex11-20180712-3362.policy.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = load_azalea_model(model_path, device)
    print(f"✅ Model loaded successfully on {device}!")
    
    # Test inference
    print("\nTesting inference...")
    dummy_board = torch.zeros((1, 11, 11), dtype=torch.long, device=device)
    
    with torch.no_grad():
        policy, value = model(dummy_board)
    
    print(f"Policy shape: {policy.shape}")  # Should be (1, 121)
    print(f"Value: {value.item():.4f}")     # Should be near 0
    
    # Speed test
    print("\nSpeed test...")
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            policy, value = model(dummy_board)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference: {avg_time*1000:.1f}ms")
    print(f"Sims possible/sec: {1/avg_time:.0f}")
    
    if avg_time < 0.03:
        print("\n✅ EXCELLENT - Fast enough!")
    elif avg_time < 0.05:
        print("\n⚠️  OK - Should work")
    else:
        print("\n❌ SLOW - Might struggle")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
