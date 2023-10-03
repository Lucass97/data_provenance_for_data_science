# Overview

DPDS (Data Provenance for Data Science) is a library capable of capturing fine-grained provenance in a preprocessing pipeline. Built on top of pandas, DPDS provides a clear and clean interface for capturing provenance without the need to invoke additional functions. The data scientist simply needs to implement the pipeline and, after executing it, can analyze the corresponding graph using Neo4j.

## Table of Contents

- [Supported Functions](#supported-functions)
- [Requirements](#requirements)
- [Installation](#installation)
- [Example](#example)
- [Demos](demos/README.md)
   - [Simple Pipelines](demos/README.md#simple_pipelines)
   - [Real World Pipelines](demos/README.md#real-world-pipelines)
      - [Census Pipeline](demos/README.md#census-pipeline)
      - [Compas Pipeline](demos/README.md#compas-pipeline)
      - [German Pipeline](demos/README.md#german-pipeline)
- [Simple Client](client/README.md)
   - [Provenance Queries](client/README.md#provenance-queries)
   - [Usage](client/README.md#usage)
      - [Command Line Arguments](client/README.md#command-line-arguments)
- [Query Tester](test/README.md#query-tester)
   - [Usage](test/README.md#usage)
      - [Command Line Arguments](test/README.md#command-line-arguments)
   - [Test All Queries Script](test/README.md#test-all-queries-script)
- [Neo4j (Docker)](#neo4j-docker)

## Supported Functions

Currently, the types of functions captured are as follows:

| Category | Function | Description | Examples |
|---|---|---|---|
| Data Reduction | *Feature Selection* | One or more features are removed. | |
| Data Reduction | *Instance Drop* | One or more records are removed. | |
| Data Augmentation | *Feature Augmentation* | One or more features are added. | |
| Data Augmentation | *Instance Generation* | One or more records are added. | |
| Space Transformation | *Dimensionality Reduction* | Features and records are added/removed. The overall number of removed features and records is greater than those added. | |
| Space Transformation | *Space Augmentation* | Features and records are added/removed. The overall number of added features and records is greater than those removed. | |
| Space Transformation | *Space Transformation* | Features and records are added/removed. In this case, there can be a reduction in dimensionality for one axis and a space augmentation for the other. | |
| Data Transformation | *Value Transformation* | The values of one or more features are transformed. | |
| Data Transformation | *Imputation* | Missing values in one or more features are filled with estimated values. | |
| Feature Manipulation | *Feature Rename* | One or more features are renamed. | |
| Data Combination | *Join* | Two or more datasets are combined based on a common attribute or key. | |
| Data Combination | *Cartesian Product* | Two datasets are combined resulting in a Cartesian product. | |

----

## Requirements

- neo4j >= 5.7.x
- pandas 1.5.0

> For further details, refer to the [requirements.txt](requirements.txt) file.

## Installation

To use the DPDS tool, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Lucass97/data_provenance_for_data_science.git
   ```

2. Navigate to the project directory:
   ```sh
   cd data_provenance_for_data_science
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
   > To create a new virtual environment (venv), use the guide at the following [link](https://docs.python.org/3/library/venv.html).

4. Install the required dependencies:
   ```shell
   pip install -r requirements.txt
   ```

## Example

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

""" Pipeline """

# Instance generation
df = df.append({'key2': 4}, ignore_index=True)
# Join
df = df.merge(right=right, on=['key1', 'key2'], how='left')
# Imputation
df = df.fillna(0)
```

Use the property ```tracker.dataframe_tracking = False``` to temporarily disable provenance tracking.

## Neo4j (Docker)

**Guide to install Neo4j via Docker**

> It is recommended to install Docker using the official guide at the following [link](https://docs.docker.com/engine/install/).
>
> To change the options related to the Neo4j Docker image, modify the file [neo4j/docker-compose.yml](neo4j/docker-compose.yml).

- Start Neo4j in the background by executing the following command:

  ```sh
  cd docker
  docker compose up -d
  ```

- To stop Neo4j, run the following command:
  ```sh
  cd docker
  docker compose down
  ```

**Default credentials:**

- `User`: `neo4j`
- `Password`: `adminadmin`

> To access the Neo4j web interface, open the following URL in your web browser:
>
> [http://localhost:7474/browser/](http://localhost:7474/browser/)