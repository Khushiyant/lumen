import sys
import os
import numpy as np

# Adjust path to your build/lib directory
sys.path.append(os.path.abspath("./build/lib"))

def run_test(backend_name):
    try:
        import lumen_py
        print(f"\n--- Verifying {backend_name.upper()} Backend ---")

        rt = lumen_py.Runtime()
        
        # Check if backend exists
        try:
            rt.set_backend(backend_name)
        except Exception:
            print(f"SKIP: {backend_name} backend not available on this system.")
            return

        if rt.current_backend() != backend_name:
            print(f"SKIP: Could not activate {backend_name} (likely not compiled or no hardware found).")
            return

        shape = [2, 4]
        buf_a = rt.alloc(shape)
        buf_b = rt.alloc(shape)
        buf_c = rt.alloc(shape)

        # Initialize data using the zero-copy views
        a_np = buf_a.data()
        b_np = buf_b.data()
        a_np[:] = np.arange(8, dtype=np.float32).reshape(shape)
        b_np[:] = 10.0

        # Execute
        rt.execute("add", [buf_a, buf_b], buf_c)
        events = rt.submit()
        for ev in events:
            ev.wait()

        # Verify results
        actual = buf_c.data()
        expected = a_np + b_np

        if np.allclose(actual, expected):
            print(f"SUCCESS: {backend_name} result matches.")
        else:
            diff = np.max(np.abs(actual - expected))
            print(f"FAILURE: {backend_name} Max Difference: {diff}")

    except Exception as e:
        print(f"Error during {backend_name} verification: {e}")

if __name__ == "__main__":
    # Test Metal
    run_test("metal")
    
    # Test CUDA
    run_test("cuda")