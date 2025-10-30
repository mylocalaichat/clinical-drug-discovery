"""
Analyze all node types in Memgraph to determine relevance for drug discovery.
"""

import os
from neo4j import GraphDatabase
from collections import defaultdict
import json

def analyze_node_types():
    """Query Memgraph to analyze all node types and their properties."""

    memgraph_uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")

    auth = None
    if memgraph_user or memgraph_password:
        auth = (memgraph_user, memgraph_password)

    driver = GraphDatabase.driver(memgraph_uri, auth=auth)

    try:
        with driver.session() as session:
            # Get all node types and their counts
            print("=" * 80)
            print("NODE TYPE ANALYSIS FOR CLINICAL DRUG DISCOVERY")
            print("=" * 80)

            result = session.run("""
                MATCH (n:Node)
                RETURN n.node_type as type, count(*) as count
                ORDER BY count DESC
            """)

            node_types = []
            total_nodes = 0
            for record in result:
                node_type = record['type']
                count = record['count']
                node_types.append((node_type, count))
                total_nodes += count

            print(f"\nTotal Nodes: {total_nodes:,}")
            print(f"Total Node Types: {len(node_types)}\n")

            print("NODE TYPE DISTRIBUTION:")
            print("-" * 80)
            for node_type, count in node_types:
                percentage = (count / total_nodes) * 100
                print(f"{node_type:30s} {count:>10,} ({percentage:>5.2f}%)")

            # Get sample nodes for each type to understand their properties
            print("\n" + "=" * 80)
            print("SAMPLE NODES BY TYPE (with properties)")
            print("=" * 80)

            analysis = {}

            for node_type, count in node_types:
                print(f"\n{node_type.upper()} (n={count:,})")
                print("-" * 80)

                # Get 5 sample nodes with all properties
                result = session.run("""
                    MATCH (n:Node {node_type: $type})
                    RETURN n.node_id as id,
                           n.node_name as name,
                           properties(n) as props
                    LIMIT 5
                """, type=node_type)

                samples = []
                for record in result:
                    node_id = record['id']
                    node_name = record['name']
                    props = record['props']
                    samples.append({
                        'id': node_id,
                        'name': node_name,
                        'properties': list(props.keys())
                    })
                    print(f"  • {node_name} (ID: {node_id})")
                    print(f"    Properties: {', '.join(props.keys())}")

                analysis[node_type] = {
                    'count': count,
                    'percentage': round((count / total_nodes) * 100, 2),
                    'samples': samples
                }

            # Get relationship patterns
            print("\n" + "=" * 80)
            print("RELATIONSHIP PATTERNS BY NODE TYPE")
            print("=" * 80)

            for node_type, _ in node_types[:10]:  # Top 10 types
                print(f"\n{node_type.upper()}:")

                # Outgoing relationships
                result = session.run("""
                    MATCH (n:Node {node_type: $type})-[r:RELATES]->(m:Node)
                    RETURN m.node_type as target_type, count(*) as count
                    ORDER BY count DESC
                    LIMIT 5
                """, type=node_type)

                outgoing = []
                for record in result:
                    outgoing.append(f"{record['target_type']} ({record['count']:,})")

                if outgoing:
                    print(f"  Connects to → {', '.join(outgoing)}")
                else:
                    print(f"  No outgoing relationships")

            # Save analysis to JSON
            with open('data/01_raw/node_type_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2)

            print("\n" + "=" * 80)
            print("Analysis saved to: data/01_raw/node_type_analysis.json")
            print("=" * 80)

            return analysis

    finally:
        driver.close()

if __name__ == "__main__":
    analyze_node_types()
