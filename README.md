# Data Provenance for Data Science
Backend with the purpose of capturing data provenance and metadata of a preprocessing Pipeline.

## Getting Started

### Requirements
- neo4j >= 4.4.x
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
python3 main.py
```

### Example
``` python
# Create dataframe df
df = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('ABCD')).astype(float)
df['D'] = np.nan
df[df < 7] = np.nan

# Create dataframe df2
df2 = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('AEFG')).astype(float)

# Create ProvenanceTracker object
tracker = ProvenanceTracker(df)

# Pipeline
tracker.df_input.append({'A': 5}, ignore_index=True)
tracker.df_input.applymap(func=lambda x: 2 if pd.isnull(x) else x // 3)
tracker.df_input.merge(right=df2, on='A', how='inner')
tracker.df_input.drop('A', axis=1, )

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