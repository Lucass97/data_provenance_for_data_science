# Data Provenance for Data Science
Backend with the purpose of capturing data provenance and metadata of a preprocessing Pipeline.

## Getting Started

### Activate venv (sh/bash)
```shell
source activate venv/bin/activate
```

### Install dependencies
```shell
pip install -r requirements.txt
```

### Run demo
```shell
python3 main.py
```

## Neo4j 

```dockerfile
version: '3.7'

services:
  neo4j:
    image: neo4j:latest
    restart: unless-stopped
    volumes:
      - neo4j_data:/data
      - /home/luca/Progetti/Data_Provenance/prov_acquisition/prov_results/:/home/luca/Progetti/Data_Provenance/prov_acquisition/prov_results/
    environment:
      - NEO4J_AUTH=neo4j/admin
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_directories_import=/
      - NEO4J_dbms_security_allow__csv__import__from__file__urls=true
    ports:
      - "7473:7473"  # https
      - "7474:7474"  # http
      - "7687:7687"  # bolt

volumes:
  neo4j_data:
```