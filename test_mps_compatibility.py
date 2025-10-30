#!/usr/bin/env python3
"""
Test script for MPS compatibility in GNN embeddings.
Tests the fix for Apple Silicon MPS device compatibility issues.
"""

import torch
import os
from pathlib import Path


def test_mps_compatibility():
    """Test MPS compatibility for GNN embeddings."""
    
    print("="*60)
    print("MPS COMPATIBILITY TEST FOR GNN EMBEDDINGS")
    print("="*60)
    
    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this device")
        return
    
    print("✓ MPS is available")
    
    # Check environment variable
    fallback_enabled = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    print(f"MPS fallback enabled: {fallback_enabled}")
    
    # Test the GNN embeddings with small graph
    print("\nTesting GNN embeddings with MPS...")
    
    try:
        from src.clinical_drug_discovery.lib.gnn_embeddings import load_graph_from_csv, train_gnn_embeddings
        
        # Load small test graph
        edges_file = "data/01_raw/primekg/nodes.csv"
        if not Path(edges_file).exists():
            print(f"❌ Test data not found: {edges_file}")
            return
        
        print("Loading small test graph...")
        data, metadata = load_graph_from_csv(
            edges_csv=edges_file,
            limit_nodes=100,  # Small test
            use_cache=True
        )
        
        print(f"✓ Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Test training with MPS
        print("\nTesting GNN training with MPS device...")
        embeddings = train_gnn_embeddings(
            data=data,
            embedding_dim=64,  # Small embedding for test
            hidden_dim=32,
            num_layers=2,
            num_epochs=2,  # Just a few epochs for test
            batch_size=32,
            learning_rate=0.01,
            device='mps'  # Force MPS
        )
        
        print("✓ Training completed successfully!")
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        print(f"✓ Embeddings device: {embeddings.device}")
        
        # Verify embeddings are on MPS
        if embeddings.device.type == 'mps':
            print("🍎 Embeddings successfully generated on MPS device")
        else:
            print(f"⚠️  Embeddings on {embeddings.device.type} (fallback occurred)")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_fallback():
    """Test device fallback behavior."""
    
    print(f"\n{'='*60}")
    print("DEVICE FALLBACK TEST")
    print(f"{'='*60}")
    
    try:
        from src.clinical_drug_discovery.lib.gnn_embeddings import load_graph_from_csv, train_gnn_embeddings
        
        # Load small test graph
        edges_file = "data/01_raw/primekg/nodes.csv"
        if not Path(edges_file).exists():
            print(f"❌ Test data not found: {edges_file}")
            return
        
        data, metadata = load_graph_from_csv(
            edges_csv=edges_file,
            limit_nodes=50,  # Very small test
            use_cache=True
        )
        
        # Test automatic device selection
        print("\nTesting automatic device selection...")
        embeddings_auto = train_gnn_embeddings(
            data=data,
            embedding_dim=32,
            hidden_dim=16,
            num_layers=1,
            num_epochs=1,
            batch_size=16,
            learning_rate=0.01,
            device=None  # Auto-select
        )
        
        print(f"✓ Auto device selection completed: {embeddings_auto.device}")
        
        # Test CPU fallback
        print("\nTesting CPU fallback...")
        embeddings_cpu = train_gnn_embeddings(
            data=data,
            embedding_dim=32,
            hidden_dim=16,
            num_layers=1,
            num_epochs=1,
            batch_size=16,
            learning_rate=0.01,
            device='cpu'  # Force CPU
        )
        
        print(f"✓ CPU training completed: {embeddings_cpu.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_environment_setup():
    """Verify the environment is properly set up for MPS."""
    
    print(f"\n{'='*60}")
    print("ENVIRONMENT VERIFICATION")
    print(f"{'='*60}")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS support
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check environment variables
    mps_fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'Not set')
    print(f"PYTORCH_ENABLE_MPS_FALLBACK: {mps_fallback}")
    
    # Test basic MPS operations
    if torch.backends.mps.is_available():
        try:
            # Test basic tensor operations
            x = torch.randn(10, 10, device='mps')
            y = torch.randn(10, 10, device='mps')
            z = torch.mm(x, y)
            print(f"✓ Basic MPS tensor operations work: {z.shape}")
            
            # Test problematic operation with fallback
            try:
                # This operation might trigger the MPS fallback
                indices = torch.tensor([0, 1, 2, 1, 3], device='mps')
                ptr = torch._convert_indices_from_coo_to_csr(indices, 4)
                print(f"✓ CSR conversion works (or fallback successful): {ptr}")
            except Exception as e:
                print(f"⚠️  CSR conversion issue (expected with MPS): {e}")
                print("✓ This is why we use CPU for sampling and MPS for training")
                
        except Exception as e:
            print(f"❌ Basic MPS operations failed: {e}")
    
    return True


if __name__ == "__main__":
    try:
        # Verify environment first
        verify_environment_setup()
        
        # Test MPS compatibility
        mps_success = test_mps_compatibility()
        
        # Test device fallback
        fallback_success = test_device_fallback()
        
        print(f"\n{'='*60}")
        if mps_success and fallback_success:
            print("✅ ALL MPS COMPATIBILITY TESTS PASSED!")
            print("🍎 Your system is ready for MPS-accelerated GNN training")
        else:
            print("⚠️  SOME TESTS FAILED")
            print("💻 Consider using CPU for training if issues persist")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()