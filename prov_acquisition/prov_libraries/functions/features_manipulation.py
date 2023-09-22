from misc.decorators import suppress_tracking, timing
from misc.utils import keys_mapping
from prov_acquisition import constants
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing(log_file=constants.FUNCTION_EXECUTION_TIMES)
def get_prov_feature_rename(tracker, dataframe_state: DataFrameState) -> dict:
    """
    Captures the provenance related to the renaming of one or more features.

    :param tracker: Provenance Tracker
    :param dataframe_state: Input and output DataFrame state.
    :return: dict - Mapping between the features of the output DataFrame and the input DataFrame.
    """

    function_name = "Feature Rename"

    df_input = dataframe_state.df_input_copy
    df_output = dataframe_state.df_output

    int_rows = df_output.index.intersection(df_input.index)

    used_entities = []
    generated_entities = []
    used_features = set()
    generated_features = set()

    output_values = df_output.to_numpy()

    hash_df_output_common_index = dataframe_state.hash_df_output_common_index.to_dict()
    hash_df_input = dataframe_state.hash_df_input.to_dict()

    feature_mapping = keys_mapping(hash_df_output_common_index, hash_df_input)

    # Iterate over the intersecting rows to find feature rename operations
    for index in int_rows:

        i = df_output.index.get_loc(index)

        for col_name1, col_name2 in feature_mapping.items():

            if col_name1 == col_name2:
                continue

            col = df_output.columns.get_loc(col_name1)
            output_value = output_values[i, col]

            used_entity = dataframe_state.index_col_to_input_entities.pop((index, col_name2), None)
            if used_entity:
                generated_entity = tracker.global_state.create_entity(value=output_value, feature_name=col_name1,
                                                                      index=index,
                                                                      instance=tracker.global_state.operation_number)
                generated_entities.append(generated_entity['id'])
                used_entities.append(used_entity['id'])
                generated_features.add(col_name1)
                used_features.add(col_name2)

                dataframe_state.index_col_to_input_entities[(index, col_name1)] = generated_entity
                tracker.global_state.create_derivation(used_ent=used_entity['id'], gen_ent=generated_entity['id'])

    if len(generated_features) > 0:
        tracker.logger.info(f' Feature rename detect: {feature_mapping}')
        act_id = tracker.global_state.create_activity(function_name, used_features=list(used_features),
                                                      description=tracker.global_state.description,
                                                      code=tracker.global_state.code,
                                                      code_line=tracker.global_state.code_line,
                                                      generated_features=list(generated_features),
                                                      tracker_id=dataframe_state.tracker_id)

        tracker.global_state.create_relation(act_id=act_id, generated=generated_entities, used=used_entities,
                                             invalidated=[],
                                             same=True)

    return feature_mapping