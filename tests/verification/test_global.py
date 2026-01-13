from src.rag_engine import RAGEngine
import logging
logging.basicConfig(level=logging.INFO)

def test_system():
    engine = RAGEngine()
    print("\n" + "="*70)
    print("--- TEST 1: Global Discovery (Community Search) ---")
    print("="*70)
    # This should trigger the GLOBAL_DISCOVERY intent
    response = engine.query("Was sind die übergeordneten technischen Themen und Architekturen in diesen Dokumenten?")
    print(f"\n[Global Answer]: {response.response}\n")
    print(f"[Source Nodes]: {len(response.source_nodes)} (should be 0 for global search)")
    print(f"[Query Plan]: {response.query_plan.sub_queries[0].intent if response.query_plan.sub_queries else 'N/A'}")

    print("\n" + "="*70)
    print("--- TEST 2: Specific Fact (Vector Search) ---")
    print("="*70)
    # This should trigger standard retrieval
    response = engine.query("Wie konfiguriere ich Qdrant in Docker?")
    print(f"\n[Local Answer]: {response.response}\n")
    print(f"[Source Nodes]: {len(response.source_nodes)}")
    print(f"[Query Plan]: {response.query_plan.sub_queries[0].intent if response.query_plan.sub_queries else 'N/A'}")
    print("\n" + "="*70)

if __name__ == "__main__":
    test_system()
