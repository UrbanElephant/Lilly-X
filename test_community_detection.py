#!/usr/bin/env python3
"""Test script for Community Detection functionality in GraphRAG system."""

import sys
import logging
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_schema_imports():
    """Test that new schema models can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Schema Imports and Validation")
    print("="*60)
    
    try:
        from src.schemas import CommunitySummary, QueryIntent
        
        # Check CommunitySummary fields
        fields = CommunitySummary.model_fields.keys()
        expected_fields = {'community_id', 'level', 'summary', 'keywords'}
        assert expected_fields.issubset(fields), f"Missing fields: {expected_fields - fields}"
        print(f"✓ CommunitySummary fields: {list(fields)}")
        
        # Check GLOBAL_DISCOVERY in QueryIntent
        assert hasattr(QueryIntent, 'GLOBAL_DISCOVERY'), "GLOBAL_DISCOVERY not found in QueryIntent"
        print(f"✓ QueryIntent.GLOBAL_DISCOVERY: {QueryIntent.GLOBAL_DISCOVERY}")
        
        # Test instantiation
        test_summary = CommunitySummary(
            community_id=1,
            level=0,
            summary="Test community summary",
            keywords=["test", "community"]
        )
        print(f"✓ CommunitySummary instantiation successful")
        print(f"  - community_id: {test_summary.community_id}")
        print(f"  - keywords: {test_summary.keywords}")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema import test FAILED: {e}")
        return False


def test_graph_operations_methods():
    """Test that GraphOperations has all required methods."""
    print("\n" + "="*60)
    print("TEST 2: GraphOperations Method Availability")
    print("="*60)
    
    try:
        from src.graph_ops import GraphOperations
        
        # Check for methods
        required_methods = [
            'run_community_detection',
            'get_nodes_in_community',
            'store_community_summary',
            'get_community_context'
        ]
        
        for method_name in required_methods:
            assert hasattr(GraphOperations, method_name), f"Method {method_name} not found"
            print(f"✓ Method '{method_name}' exists")
        
        # Check method signatures
        import inspect
        
        # Check run_community_detection signature
        sig = inspect.signature(GraphOperations.run_community_detection)
        params = list(sig.parameters.keys())
        assert 'algorithm' in params, "Missing 'algorithm' parameter"
        assert 'write_property' in params, "Missing 'write_property' parameter"
        print(f"✓ run_community_detection signature: {params}")
        
        # Check get_community_context signature
        sig = inspect.signature(GraphOperations.get_community_context)
        params = list(sig.parameters.keys())
        assert 'keywords' in params, "Missing 'keywords' parameter"
        assert 'top_k' in params, "Missing 'top_k' parameter"
        print(f"✓ get_community_context signature: {params}")
        
        return True
        
    except Exception as e:
        print(f"✗ GraphOperations method test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neo4j_connection():
    """Test Neo4j connection and GDS availability."""
    print("\n" + "="*60)
    print("TEST 3: Neo4j Connection and GDS Availability")
    print("="*60)
    
    try:
        from src.graph_database import get_neo4j_driver
        
        driver = get_neo4j_driver()
        print(f"✓ Neo4j driver connection successful")
        
        # Check for GDS
        with driver.session() as session:
            result = session.run(
                "CALL dbms.procedures() YIELD name "
                "WHERE name STARTS WITH 'gds' "
                "RETURN count(name) as gds_count"
            )
            gds_count = result.single()["gds_count"]
            
            if gds_count > 0:
                print(f"✓ Neo4j GDS plugin detected ({gds_count} procedures)")
                
                # List some key GDS procedures
                result = session.run(
                    "CALL dbms.procedures() YIELD name "
                    "WHERE name STARTS WITH 'gds.leiden' OR name STARTS WITH 'gds.louvain' "
                    "RETURN name ORDER BY name LIMIT 5"
                )
                procedures = [record["name"] for record in result]
                print(f"  Available algorithms: {procedures}")
            else:
                print(f"⚠ Neo4j GDS plugin NOT detected")
                print(f"  Community detection will not work without GDS")
                print(f"  Install: https://neo4j.com/docs/graph-data-science/current/installation/")
                return False
            
            # Check for Entity nodes
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]
            print(f"✓ Entity nodes in graph: {entity_count}")
            
            if entity_count == 0:
                print(f"  ⚠ Warning: No Entity nodes found. Run ingestion first.")
        
        return True
        
    except Exception as e:
        print(f"✗ Neo4j connection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_community_detection_dry_run():
    """Test community detection without actually running it (if graph is empty)."""
    print("\n" + "="*60)
    print("TEST 4: Community Detection Dry Run")
    print("="*60)
    
    try:
        from src.graph_ops import GraphOperations
        
        ops = GraphOperations()
        print(f"✓ GraphOperations instance created")
        
        # Check if we have entities to work with
        from src.graph_database import get_neo4j_driver
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]
            
            if entity_count > 10:
                print(f"✓ Graph has {entity_count} entities - ready for community detection")
                print(f"  To run community detection, execute:")
                print(f"    from src.graph_ops import GraphOperations")
                print(f"    ops = GraphOperations()")
                print(f"    stats = ops.run_community_detection(algorithm='leiden')")
                print(f"    print(stats)")
            else:
                print(f"  ⚠ Graph has only {entity_count} entities")
                print(f"  Skipping actual community detection test")
                print(f"  Run data ingestion first to populate the graph")
        
        # Test instantiating a CommunitySummary
        from src.schemas import CommunitySummary
        test_summary = CommunitySummary(
            community_id=999,
            level=0,
            summary="Test community for ML frameworks",
            keywords=["tensorflow", "pytorch", "machine learning"]
        )
        print(f"✓ Created test CommunitySummary: {test_summary.summary}")
        
        # Test get_community_context with empty keywords
        results = ops.get_community_context(keywords=[], top_k=5)
        assert results == [], "Expected empty results for empty keywords"
        print(f"✓ get_community_context handles empty keywords correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Community detection dry run FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GraphRAG Community Detection - Verification Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Schema Imports", test_schema_imports()))
    results.append(("GraphOperations Methods", test_graph_operations_methods()))
    results.append(("Neo4j Connection", test_neo4j_connection()))
    results.append(("Community Detection Dry Run", test_community_detection_dry_run()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("="*60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
