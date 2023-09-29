#!/bin/bash

# Set variables for Neo4j configuration
URI="bolt://localhost"
USER="neo4j"
PWD="adminadmin"
LIMIT=3

# Loop through all available commands
for COMMAND in "all-transformations" "why-provenance" "how-provenance" \
               "dataset-level-feature-operation" "record-operation" \
               "record-invalidation" "item-invalidation" \
               "item-level-feature-operation" "item-history" \
               "record-history" "feature-invalidation"; do

    # Execute the corresponding command
    python3 query_tester.py --uri "$URI" --user "$USER" --pwd "$PWD" --limit "$LIMIT" "$COMMAND"

    echo "Executed command: $COMMAND"

done