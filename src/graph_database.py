"""Neo4j graph database client management."""

from typing import Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import settings


# Singleton instance
_neo4j_driver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Get or create a singleton Neo4j Driver instance.

    Returns:
        Driver: Configured Neo4j driver connected to the graph database.

    Raises:
        ConnectionError: If unable to connect to Neo4j.
        AuthError: If authentication fails.
    """
    global _neo4j_driver

    if _neo4j_driver is None:
        try:
            _neo4j_driver = GraphDatabase.driver(
                settings.neo4j_url,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
            # Verify connection
            _neo4j_driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {settings.neo4j_url}")
        except AuthError as e:
            raise AuthError(
                f"Authentication failed for Neo4j at {settings.neo4j_url}. "
                f"Check NEO4J_USER and NEO4J_PASSWORD in .env"
            ) from e
        except ServiceUnavailable as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j at {settings.neo4j_url}. "
                f"Ensure Neo4j container is running."
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Unexpected error connecting to Neo4j: {e}"
            ) from e

    return _neo4j_driver


def close_neo4j_driver() -> None:
    """Close the Neo4j driver connection and reset the singleton."""
    global _neo4j_driver

    if _neo4j_driver is not None:
        try:
            _neo4j_driver.close()
            print("✓ Neo4j driver connection closed")
        except Exception as e:
            print(f"⚠ Error closing Neo4j driver: {e}")
        finally:
            _neo4j_driver = None


if __name__ == "__main__":
    """Test Neo4j connection."""
    print("=" * 50)
    print("Lilly-X - Neo4j Connection Test")
    print("=" * 50)
    print()
    
    try:
        # Test connection
        print(f"📡 Connecting to Neo4j at {settings.neo4j_url}...")
        print(f"   User: {settings.neo4j_user}")
        driver = get_neo4j_driver()
        print()
        
        # Get database info
        print("📋 Fetching database information...")
        with driver.session() as session:
            # Get Neo4j version
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            record = result.single()
            print(f"✓ Neo4j Version: {record['versions'][0]}")
            print(f"✓ Edition: {record['edition']}")
            print()
            
            # Check for existing nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✓ Total nodes in database: {count}")
            print()
            
            # Check for APOC
            result = session.run(
                "CALL dbms.procedures() YIELD name "
                "WHERE name STARTS WITH 'apoc' "
                "RETURN count(name) as count"
            )
            apoc_count = result.single()["count"]
            if apoc_count > 0:
                print(f"✓ APOC plugin loaded ({apoc_count} procedures)")
            else:
                print("⚠ APOC plugin not detected")
            
            # Check for GDS
            result = session.run(
                "CALL dbms.procedures() YIELD name "
                "WHERE name STARTS WITH 'gds' "
                "RETURN count(name) as count"
            )
            gds_count = result.single()["count"]
            if gds_count > 0:
                print(f"✓ Graph Data Science plugin loaded ({gds_count} procedures)")
            else:
                print("⚠ Graph Data Science plugin not detected")
            print()
        
        print("=" * 50)
        print("✅ Connection test SUCCESSFUL!")
        print("=" * 50)
        print()
        print("Configuration:")
        print(f"  - Neo4j URL: {settings.neo4j_url}")
        print(f"  - Neo4j User: {settings.neo4j_user}")
        print(f"  - Database: Ready for graph operations")
        print()
        
    except AuthError as e:
        print()
        print("=" * 50)
        print("❌ Authentication FAILED!")
        print("=" * 50)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check NEO4J_USER and NEO4J_PASSWORD in .env")
        print("  2. Default credentials: neo4j/password")
        print("  3. Update .env from .env.template if needed")
        exit(1)
    except ConnectionError as e:
        print()
        print("=" * 50)
        print("❌ Connection test FAILED!")
        print("=" * 50)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure Neo4j container is running: podman ps")
        print("  2. Start containers: podman-compose up -d")
        print("  3. Check Neo4j is accessible: curl http://127.0.0.1:7474")
        print("  4. Check firewall settings")
        exit(1)
    except Exception as e:
        print()
        print("=" * 50)
        print("❌ Unexpected error!")
        print("=" * 50)
        print(f"Error: {e}")
        exit(1)
    finally:
        close_neo4j_driver()
