from typing import List

from misc.decorators import suppress_tracking, timing
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing
def get_prov_from_aggregate(tracker, dataframe_state: DataFrameState, remaining_features: List[str]) -> None:
    """
    Captures the provenance related to an aggregate operation.
    TODO
    """

    function_name = 'Aggregate'

    df_input = dataframe_state.df_input_copy
    df_output = dataframe_state.df_output

    grouped_features = df_input.columns.difference(remaining_features)

    generated_entities = []
    used_entities = []
    invalidated_entities = []

    # Per ogni coppia value1, col1 (di features), punta ad un dizionario con coppia chiave col2
    # e che come valore ha un lista contente le entità.
    value_col_to_entities = {}
    for index, row in df_input.iterrows():
        for col in grouped_features:
            idx_value = row[col]
            entity = dataframe_state.index_col_to_input_entities.get((index, col))
            if (idx_value, col) not in value_col_to_entities:
                value_col_to_entities[(idx_value, col)] = {}
            if entity:
                for col2 in remaining_features:
                    entity = dataframe_state.index_col_to_input_entities.get((index, col2))
                    if col2 not in value_col_to_entities[(idx_value, col)]:
                        value_col_to_entities[(idx_value, col)][col2] = []
                    value_col_to_entities[(idx_value, col)][col2].append(entity)

    def get_provenance(row):
        comb = list(zip(row.name, grouped_features))

        for col2 in remaining_features:

            generated_entity = tracker.global_state.create_entity(value=row[col2], feature_name=col2, index=row.name,
                                                                  instance=tracker.global_state.operation_number)

            for idx_value, col in comb:

                entities = value_col_to_entities.get((idx_value, col), {})[col2]
                generated_entities.append(generated_entity['id'])

                for used_entity in entities:
                    tracker.global_state.create_derivation(gen_ent=generated_entity['id'], used_ent=used_entity['id'])
                    used_entities.append(used_entity['id'])

        return row

    df_output.apply(get_provenance, axis=1)

    for keys in dataframe_state.index_col_to_input_entities:
        invalidated_entities.append(dataframe_state.index_col_to_input_entities.get(keys)['id'])

    dataframe_state.index_col_to_input_entities = {}

    if len(used_entities) > 0:
        tracker.logger.info(f'Aggregate transformation detect')

        act_id = tracker.global_state.create_activity(function_name=function_name, used_features=list(grouped_features),
                                                      description=tracker.global_state.description,
                                                      code=tracker.global_state.code,
                                                      code_line=tracker.global_state.code_line,
                                                      tracker_id=dataframe_state.tracker_id)

        tracker.global_state.create_relation(act_id=act_id, generated=generated_entities,
                                             used=used_entities, invalidated=invalidated_entities, same=False)
