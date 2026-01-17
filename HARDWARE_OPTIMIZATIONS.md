# MAX+ 395 Hardware-Optimierungen für LLIX

## Übersicht

Dieses Dokument beschreibt die Hardware-spezifischen Optimierungen für das LLIX RAG-System auf dem AMD Ryzen AI MAX+ 395 System.

## System-Spezifikationen

- **CPU**: AMD Ryzen AI MAX+ 395 (16 Cores / 32 Threads)
  - Base: 3.0 GHz
  - Boost: 5.1 GHz
- **RAM**: 128GB LPDDR5x
  - 96GB verfügbar für System
  - 32GB reserviert als vRAM für iGPU (Radeon 8060S)
- **OS**: Fedora 42 (Bleeding Edge)
- **Python**: 3.12.12 (kritisch für Kompatibilität)
- **Container Runtime**: Podman

## Hardware-Validierung

### Validierungsskript

Das Skript [check_hardware.py](file:///home/gerrit/Antigravity/LLIX/check_hardware.py) überprüft:

✅ **Python Version** - Stellt sicher, dass Python 3.12.x verwendet wird (nicht 3.14)  
✅ **CPU-Kerne** - Erkennt 16 physische / 32 logische Kerne  
✅ **RAM** - Validiert >100GB High-Memory System  
✅ **Torch** - Verifiziert Intra-op Thread-Konfiguration

### Ausführung

```bash
./venv/bin/python check_hardware.py
```

### Beispiel-Ausgabe

```
==================================================
🚀 LLIX HARDWARE & ENVIRONMENT VALIDATION
==================================================
Python Version:  3.12.12         ✅ OK
CPU Kerne:       16 Physisch / 32 Threads
Gesamt-RAM:      94.07 GB
Torch Version:   2.9.1+cpu
Intra-op Threads: 16 (Paralleles Rechnen)
==================================================
```

## Podman Container-Optimierungen

### Qdrant (Vektor-Datenbank)

**Container**: `llix_vector_db`  
**Image**: `qdrant/qdrant:latest`

#### Kritische Optimierungen

```yaml
environment:
  # MMAP ausschalten = Vektoren im RAM halten (128GB verfügbar)
  - QDRANT__STORAGE__PERFORMANCE__MMAP_THRESHOLD_KB=0
  
  # Max Vektor-Größe an 32GB vRAM anpassen
  - QDRANT__STORAGE__PERFORMANCE__MAX_VECTORS_SIZE_GB=32
  
  # Nutze 16 von 32 Threads für Index-Optimierung
  - QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=16
  
  # HNSW-Parameter für schnelle Suche
  - QDRANT__STORAGE__HNSW__M=16
  - QDRANT__STORAGE__HNSW__EF_CONSTRUCT=100
```

#### Performance-Implikationen

| Parameter | Wert | Effekt |
|-----------|------|--------|
| `MMAP_THRESHOLD_KB=0` | 0 | Gesamter Index im RAM → Minimale Latenz |
| `MAX_VECTORS_SIZE_GB` | 32 | Nutzt verfügbares vRAM voll aus |
| `MAX_OPTIMIZATION_THREADS` | 16 | Parallele Index-Optimierung (50% CPU) |
| `HNSW M` | 16 | Balanciert Genauigkeit vs. Geschwindigkeit |

**Erwartete Speichernutzung**: 20-40GB RAM bei typischen Dokumenten-Korpora

### Neo4j (Graph-Datenbank)

**Container**: `neo4j`  
**Image**: `neo4j:5.15`

#### Optimierungen

```yaml
environment:
  # Heap-Größe für Graph-Operationen
  - NEO4J_dbms_memory_heap_max__size=4G
  
  # Pagecache für häufig genutzte Knoten/Kanten
  - NEO4J_dbms_memory_pagecache_size=8G
  
  # GDS + APOC Plugins
  - NEO4J_PLUGINS=["apoc","graph-data-science"]
```

#### Performance-Implikationen

- **4G Heap**: Ausreichend für komplexe Cypher-Queries und Aggregationen
- **8G Pagecache**: Hält die gesamte Graph-Struktur im RAM (bei mittelgroßen Graphen)
- **GDS Plugin**: Ermöglicht PageRank, Community Detection, Similarity Projections

## Python 3.12 Requirement

### Warum Python 3.12?

Fedora 42 nutzt standardmäßig:
- **Python 3.14** (development version)
- **GCC 15** (cutting-edge compiler)

Diese Kombination führt zu **Build-Fehlern** bei:
- `torch` (CUDA/CPU Binaries)
- `pandas` (C-Extensions)
- `numpy` (BLAS/LAPACK Bindings)

### Lösung

**Explizite Python 3.12 Nutzung**:

```bash
# Virtual Environment erstellen
/usr/bin/python3.12 -m venv venv

# Aktivieren
source venv/bin/activate

# Verifizieren
python --version  # Python 3.12.12
```

### Vorteile

✅ Pre-compiled Wheels verfügbar (keine Compilation nötig)  
✅ Kompatibilität mit stabilen Bibliotheken  
✅ Keine GCC 15 Build-Fehler  
✅ Schnellere `pip install` Zeit

## Empfohlene Konfigurationen

### Ingestion Pipeline

Für maximale Ingestion-Geschwindigkeit in [src/config.py](file:///home/gerrit/Antigravity/LLIX/src/config.py):

```python
# Nutze alle 32 Threads für Parallel Processing
num_workers=20  # 20 Worker für parallele Transformation

# Batch-Size an RAM anpassen
batch_size=64   # 64 Chunks pro Batch (bei 128GB RAM)
```

### Environment Variables (.env)

```bash
# Hardware-optimierte Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
BATCH_SIZE=64
TOP_K=3

# Two-Stage Retrieval
TOP_K_RETRIEVAL=50  # Erhöht von 25 (mehr RAM verfügbar)
TOP_K_FINAL=5
```

## Monitoring & Validierung

### Container-Status prüfen

```bash
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Resource Usage

```bash
# Qdrant Memory
podman stats llix_vector_db --no-stream

# Neo4j Memory
podman stats neo4j --no-stream
```

### Hardware-Check wiederholen

```bash
cd /home/gerrit/Antigravity/LLIX
./venv/bin/python check_hardware.py
```

## Troubleshooting

### Container startet nicht

**Symptom**: `bind: address already in use`

**Lösung**:
```bash
podman stop qdrant neo4j llix_vector_db
podman rm qdrant neo4j llix_vector_db
podman-compose up -d
```

### Compilation-Fehler bei pip install

**Symptom**: `error: invalid command 'bdist_wheel'`

**Lösung**: Falsche Python-Version aktiv!
```bash
# Prüfen
python --version  # MUSS 3.12.x zeigen

# Fix
deactivate
rm -rf venv
/usr/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Qdrant läuft nicht aus dem RAM

**Symptom**: Langsame Vektor-Suche

**Prüfung**:
```bash
podman logs llix_vector_db | grep mmap
```

**Erwartete Ausgabe**: Sollte KEIN "mmap" erwähnen

**Fix**: Stelle sicher, dass `MMAP_THRESHOLD_KB=0` in `compose.yaml` gesetzt ist

## Nächste Schritte

1. **Dokumente ingestieren**:
   ```bash
   # PDFs nach ./data/docs/ kopieren
   python -m src.ingest
   ```

2. **UI starten**:
   ```bash
   streamlit run src/app.py
   ```

3. **Performance testen**:
   - Nutze `bash verify_setup.sh` für Health Checks
   - Beobachte Container-Logs mit `podman logs -f llix_vector_db`

## Zusammenfassung

Das System ist jetzt optimiert für:
- ✅ **Hohe Parallelität**: 16 Optimization Threads (Qdrant)
- ✅ **Minimale Latenz**: Vektoren im RAM (MMAP=0)
- ✅ **Große Graphen**: 4G Heap + 8G Pagecache (Neo4j)
- ✅ **Stabile Dependencies**: Python 3.12.12
- ✅ **Hardware-Validierung**: Automatisches Check-Skript

---

**Erstellt**: 2026-01-17  
**System**: AMD Ryzen AI MAX+ 395 @ Fedora 42  
**Python**: 3.12.12  
**Container Runtime**: Podman
