# Neo4j GDS Plugin Setup Guide

## ✅ Current Status

Your `compose.yaml` is **already correctly configured** with the GDS plugin:

```yaml
neo4j:
  environment:
    - NEO4J_PLUGINS=["apoc","graph-data-science"]  # ✅ GDS enabled
    - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*  # ✅ Procedures allowed
```

---

## 🔍 Verification

### Quick Check

Run the verification script:

```bash
bash verify_gds.sh
```

**Expected Output:**
```
✅ Neo4j container is running
✅ Neo4j HTTP endpoint is accessible
✅ GDS plugin is installed (142 procedures available)

Key GDS procedures found:
  - gds.leiden.stream
  - gds.leiden.write
  - gds.louvain.stream
  - gds.louvain.write

✅ Community detection algorithms are available!
```

### Manual Verification

If you prefer to check manually:

```bash
# 1. Check Neo4j container is running
podman ps | grep neo4j

# 2. Test GDS via Python
python3 -c "
from src.graph_database import get_neo4j_driver
driver = get_neo4j_driver()
with driver.session() as session:
    result = session.run('CALL dbms.procedures() YIELD name WHERE name STARTS WITH \"gds\" RETURN count(name) as count')
    print(f'GDS procedures: {result.single()[\"count\"]}')
"
```

---

## 🔧 Troubleshooting

### Problem 1: GDS Plugin Not Found

**Symptoms:**
```
RuntimeError: Neo4j Graph Data Science (GDS) library is not available.
```

**Solutions:**

1. **Check compose.yaml configuration:**
   ```bash
   grep -A 5 "NEO4J_PLUGINS" compose.yaml
   ```
   
   Should show:
   ```yaml
   - NEO4J_PLUGINS=["apoc","graph-data-science"]
   ```

2. **Recreate Neo4j container:**
   ```bash
   # Stop and remove old container
   podman-compose down neo4j
   
   # Remove data volume (WARNING: deletes all graph data!)
   rm -rf neo4j_data/
   
   # Start fresh
   podman-compose up -d neo4j
   
   # Wait for startup (60 seconds)
   sleep 60
   
   # Verify
   bash verify_gds.sh
   ```

3. **Check container logs:**
   ```bash
   podman logs neo4j | grep -i gds
   ```
   
   Should show:
   ```
   INFO  Started GDS library
   INFO  Loaded Graph Data Science plugin
   ```

---

### Problem 2: Plugin Loaded But Algorithms Not Available

**Symptoms:**
```
GDS procedures: 0
```

**Solutions:**

1. **Check security configuration:**
   ```bash
   podman exec neo4j cat /var/lib/neo4j/conf/neo4j.conf | grep unrestricted
   ```
   
   Should include:
   ```
   dbms.security.procedures.unrestricted=apoc.*,gds.*
   ```

2. **Restart with explicit configuration:**
   ```bash
   podman-compose restart neo4j
   ```

---

### Problem 3: Neo4j Version Compatibility

**Issue:** GDS plugin might not be compatible with all Neo4j versions.

**Current Setup:**
- Neo4j: `5.15` (from your compose.yaml)
- GDS: Installed via `NEO4J_PLUGINS` environment variable

**Recommended Versions:**
- Neo4j 5.15+ with GDS 2.5+
- Neo4j 5.x series is fully compatible

**Check versions:**
```bash
# Neo4j version
podman exec neo4j cypher-shell -u neo4j -p password "CALL dbms.components() YIELD name, versions RETURN name, versions"

# GDS version
podman exec neo4j cypher-shell -u neo4j -p password "RETURN gds.version() AS version"
```

---

## 🚀 First-Time Setup

If you're setting up Neo4j with GDS for the first time:

### 1. Update compose.yaml

```yaml
neo4j:
  image: neo4j:5.15
  environment:
    # ... existing config ...
    - NEO4J_PLUGINS=["apoc","graph-data-science"]
    - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
    - NEO4J_dbms_memory_heap_max__size=2G  # Recommended for GDS
```

### 2. Start Container

```bash
podman-compose up -d neo4j
```

### 3. Wait for Startup

```bash
# GDS plugin download and initialization can take 60-90 seconds
sleep 90
```

### 4. Verify Installation

```bash
bash verify_gds.sh
```

---

## 📝 Performance Tuning for GDS

For better community detection performance, consider these Neo4j settings:

```yaml
neo4j:
  environment:
    # Increase heap for large graphs
    - NEO4J_dbms_memory_heap_max__size=4G
    
    # Increase page cache for better GDS performance
    - NEO4J_dbms_memory_pagecache_size=2G
    
    # GDS-specific tuning
    - NEO4J_gds_enterprise=false  # Use community edition features
```

**Recommended Settings by Graph Size:**

| Nodes | Heap Size | Page Cache |
|-------|-----------|------------|
| < 10K | 2G | 1G |
| 10K-100K | 4G | 2G |
| 100K-1M | 8G | 4G |
| > 1M | 16G | 8G |

---

## 🔗 Useful Commands

### Check Container Status
```bash
podman ps -a | grep neo4j
```

### View Logs
```bash
podman logs -f neo4j
```

### Restart Neo4j
```bash
podman-compose restart neo4j
```

### Access Neo4j Shell
```bash
podman exec -it neo4j cypher-shell -u neo4j -p password
```

### Test GDS in Neo4j Shell
```cypher
// Check GDS version
RETURN gds.version();

// List GDS procedures
CALL gds.list();

// Test simple algorithm (on empty graph)
CALL gds.graph.project('test', '*', '*');
CALL gds.pageRank.stream('test');
CALL gds.graph.drop('test');
```

---

## 📚 References

- [Neo4j GDS Documentation](https://neo4j.com/docs/graph-data-science/current/)
- [GDS Installation Guide](https://neo4j.com/docs/graph-data-science/current/installation/)
- [Leiden Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
- [Louvain Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/louvain/)

---

## ✅ Quick Checklist

Before running community detection:

- [ ] `compose.yaml` has `NEO4J_PLUGINS=["apoc","graph-data-science"]`
- [ ] Neo4j container is running: `podman ps | grep neo4j`
- [ ] GDS verification passes: `bash verify_gds.sh`
- [ ] Graph has entities: `MATCH (e:Entity) RETURN count(e)`
- [ ] Ready to run: `python3 run_community_summarization.py`
