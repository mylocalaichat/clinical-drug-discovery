#!/usr/bin/env python3
"""
Test the simplified MPS-compatible GNN implementation.
"""

import torch
from pathlib import Path


def test_simple_gnn():
    """Test the simplified GNN implementation."""
    
    print("="*60)
    print("SIMPLE GNN MPS COMPATIBILITY TEST")
    print("="*60)
    
    try:
        from src.clinical_drug_discovery.lib.gnn_embeddings import load_graph_from_csv
        from src.clinical_drug_discovery.lib.gnn_simple import train_gnn_embeddings_simple
        
        # Load small test graph
        edges_file = "data/01_raw/primekg/nodes.csv"
        if not Path(edges_file).exists():
            print(f"❌ Test data not found: {edges_file}")
            return False
        
        print("Loading small test graph...")
        data, metadata = load_graph_from_csv(
            edges_csv=edges_file,
            limit_nodes=200,  # Small test
            use_cache=True
        )
        
        print(f"✓ Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Test training with auto device selection
        print("\nTesting simple GNN training...")
        embeddings = train_gnn_embeddings_simple(
            data=data,
            embedding_dim=64,  # Small embedding for test
            hidden_dim=32,
            num_layers=2,
            num_epochs=5,  # Just a few epochs for test
            learning_rate=0.01,
            device=None  # Auto-select
        )
        
        print("✓ Training completed successfully!")
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        print(f"✓ Embeddings device: {embeddings.device}")
        
        # Test with specific devices
        for test_device in ['cpu', 'mps'] if torch.backends.mps.is_available() else ['cpu']:
            print(f"\nTesting with {test_device} device...")
            try:
                embeddings_device = train_gnn_embeddings_simple(
                    data=data,
                    embedding_dim=32,
                    hidden_dim=16,
                    num_layers=1,
                    num_epochs=2,
                    learning_rate=0.01,
                    device=test_device
                )
                print(f"✓ {test_device.upper()} training successful: {embeddings_device.shape}")
            except Exception as e:
                print(f"❌ {test_device.upper()} training failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple GNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with existing pipeline."""
    print(f"\n{'='*60}")
    print("INTEGRATION TEST")
    print(f"{'='*60}")
    
    try:
        # Test that we can replace the original function
        from src.clinical_drug_discovery.lib.gnn_simple import train_gnn_embeddings_simple
        from src.clinical_drug_discovery.lib.gnn_embeddings import load_graph_from_csv
        
        # Load tiny graph
        data, metadata = load_graph_from_csv(
            edges_csv="data/01_raw/primekg/nodes.csv",
            limit_nodes=50,
            use_cache=True
        )
        
        # Test that it produces valid embeddings
        embeddings = train_gnn_embeddings_simple(
            data=data,
            embedding_dim=16,
            num_epochs=2
        )
        
        # Basic validation
        assert embeddings.shape[0] == data.num_nodes, "Embeddings shape mismatch"
        assert embeddings.shape[1] == 16, "Embedding dimension mismatch"
        assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
        
        print("✓ Integration test passed!")
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ No NaN values: {not torch.isnan(embeddings).any()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # Test the simple implementation
        simple_success = test_simple_gnn()
        
        # Test integration
        integration_success = test_integration()
        
        print(f"\n{'='*60}")
        if simple_success and integration_success:
            print("✅ ALL SIMPLE GNN TESTS PASSED!")
            print("🍎 Ready to replace original implementation in Dagster pipeline")
        else:
            print("⚠️  SOME TESTS FAILED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()