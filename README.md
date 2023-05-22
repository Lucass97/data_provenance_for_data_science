# Data Provenance for Data Science
Backend with the purpose of capturing data provenance and metadata of a preprocessing Pipeline.

## Getting Started

### Requirements
- neo4j >= 5.7.x
- pandas 1.5.0

> For further details, refer to the [requirements.txt](requirements.txt) file.

### Activate venv (sh/bash)

> To create a new virtual environment (venv), use the guide at the following [link](https://docs.python.org/3/library/venv.html)

```shell
source activate venv/bin/activate
```

### Install dependencies
```shell
pip install -r requirements.txt
```

### Run demo
```shell
cd demos
python3 demo_shell_numerical.py
```

### Example
```python

df = pd.DataFrame({'key1': [0, 0, 1, 2, 0],
                    'key2': [0, np.nan, 0, 1, 0],
                    'A': [0, 1, 2, 3, 4],
                    'B': [0, 1, 2, 3, 4]
                    })
right = pd.DataFrame({'key1': [0, np.nan, 1, 2 ],
                        'key2': [0, 4, 0, 0],
                        'A': [0, 1, 2, 3],
                        'D': [0, np.nan, 2, 3],
                        'C': [0, 1, 2, 3]})
df2 = pd.DataFrame({'key1': [0, 5, 7, 10, 1],
                    'key2': [0, 4, 2, 1, 0],
                    'E': [1, 1, 2, 3, 9],
                    'F': [0, 1, 2, 3, 4]
                    })

# Create provenance tracker
tracker = ProvenanceTracker()

# Create tracked dataframe
df, right, df2 = tracker.subscribe([df, right, df2])

# Pipeline

# Instance generation
df = df.append({'key2': 4}, ignore_index=True)
# Join
df = df.merge(right=right, on=['key1', 'key2'], how='left')
# Imputation
df = df.fillna(0)
```

Use the property ```tracker.dataframe_tracking = False``` to temporarily disable provenance tracking.

### Install Neo4j via Docker

> It is recommended to install Docker using the official guide at the following [link](https://docs.docker.com/engine/install/)

> To change the options related to the Neo4j Docker image, modify the file [neo4j/docker-compose.yml](neo4j/docker-compose.yml)

Start Neo4j in background:

```sh
cd neo4j
docker compose up -d
```

Stop Neo4j:

```sh
cd neo4j
docker compose down
```

Default credentials:
- **User**: *neo4j*
- **Password**: *admin*

To access the Neo4j web interface:
- [http://localhost:7474/browser/](http://localhost:7474/browser/)