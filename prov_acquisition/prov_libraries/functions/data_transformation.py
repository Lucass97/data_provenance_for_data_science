import pandas as pd

from misc.decorators import timing, suppress_tracking
from prov_acquisition import constants
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing(log_file=constants.FUNCTION_EXECUTION_TIMES)
def get_prov_value_change(tracker, dataframe_state: DataFrameState, extra_used_features: set = None) -> None:
    """
    Captures the provenance related to a change in the values of the DataFrame.
    The type of generated activity can be of two types:
    Value Transformation: generic case.
    Imputation: the DataFrame column has undergone a replacement of null values.

    :param tracker: Provenance Tracker
    :param dataframe_state: Input and output DataFrame state.
    :param extra_used_features: Extra features used in the input. Indicates additional features that contribute to the generation of new entities.
    :return: None
    """

    tracker.logger.info(f' Check for value change...')

    function_name1 = "Value Transformation"
    function_name2 = "Imputation"

    df_input = dataframe_state.df_input_copy
    df_output = dataframe_state.df_output

    int_columns = df_output.columns.intersection(df_input.columns)

    generated_entities = []
    used_entities = []
    extra_used_entities = []
    imp_cols = set()
    trans_cols = set()
    values_output = df_output[int_columns].to_numpy()

    if extra_used_features is None:
        used_features = set()

    for index in df_output.index:

        i = df_output.index.get_loc(index)

        for col_name in int_columns:

            col = int_columns.get_loc(col_name)
            new_value = values_output[i][col]

            used_entity = dataframe_state.index_col_to_input_entities.get((index, col_name), None)

            if used_entity is None:
                continue

            old_value = used_entity['value']

            if new_value != old_value:
                if pd.isnull(old_value) and pd.isnull(new_value):
                    continue
                elif pd.isnull(old_value):
                    imp_cols.add(col_name)
                else:
                    trans_cols.add(col_name)

                entity = tracker.global_state.create_entity(value=new_value, feature_name=col_name, index=index,
                                                            instance=tracker.global_state.operation_number)

                dataframe_state.index_col_to_input_entities[(index, col_name)] = entity

                generated_entities.append(entity['id'])
                used_entities.append(used_entity['id'])

                tracker.global_state.create_derivation(used_ent=used_entity['id'], gen_ent=entity['id'])

                # Extra features to add
                for used_feature in used_features:
                    used_entity = dataframe_state.index_col_to_input_entities.get((index, used_feature), None)

                    if used_entity is None:
                        continue

                    extra_used_entities.append(used_entity)
                    tracker.global_state.create_derivation(used_ent=used_entity['id'], gen_ent=entity['id'])

    imp_cols = imp_cols.difference(trans_cols)

    if len(imp_cols) > 0:
        tracker.logger.info(f' Imputation detect on {imp_cols} columns')
        act_id = tracker.global_state.create_activity(function_name=function_name2, used_features=list(imp_cols),
                                                      description=tracker.global_state.description,
                                                      code=tracker.global_state.code,
                                                      code_line=tracker.global_state.code_line,
                                                      tracker_id=dataframe_state.tracker_id)
        tracker.global_state.create_relation(act_id=act_id, generated=generated_entities, used=None,
                                             invalidated=None, same=True)

    if len(trans_cols) > 0:
        tracker.logger.info(f' Value transformation detect on {trans_cols} columns')
        trans_cols = trans_cols.union(used_features)
        extra_used_entities.extend(used_entities)
        act_id = tracker.global_state.create_activity(function_name=function_name1, used_features=list(trans_cols),
                                                      description=tracker.global_state.description,
                                                      code=tracker.global_state.code,
                                                      code_line=tracker.global_state.code_line,
                                                      tracker_id=dataframe_state.tracker_id)
        tracker.global_state.create_relation(act_id=act_id, generated=generated_entities,
                                             used=used_entities if len(
                                                 extra_used_entities) == 0 else extra_used_entities,
                                             invalidated=None, same=len(extra_used_entities) == len(used_entities))
    else:
        tracker.logger.info(f' Value transformation and Imputation not detected')


