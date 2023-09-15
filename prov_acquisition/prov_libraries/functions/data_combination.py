from typing import Set, Tuple

from misc.decorators import suppress_tracking, timing
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing
def get_prov_join(tracker, dataframe_state_left: DataFrameState, dataframe_state_right: DataFrameState, how: str,
                  left_keys: Set[str], right_keys: Set[str], suffixes: Tuple[str],
                  _merge_feature: bool = False) -> None:
    """
    Captures the provenance related to the join operation.
    Known issues to address: The merge operation can convert integer columns to float columns if they contain null values.
    Changing the type changes the hash of the row, resulting in missing corresponding indices.

    :param tracker: Provenance Tracker
    :param dataframe_state_left: The first input dataframe state.
    :param dataframe_state_right: The second input dataframe state.
    :param how: Type of join.
    :param left_keys: Set of keys used for the left dataframe.
    :param right_keys: Set of keys used for the right dataframe.
    :param suffixes: Suffixes for common keys.
    :param _merge_feature: Indicates if the merge feature has been previously generated for provenance.

    """

    function_name = "Join"

    if how == 'left':
        function_name = 'Left Join'

    if how == 'right':
        function_name = 'Right Join'

    if how == 'cross':
        function_name = 'Cartesian Product'
        left_keys = set()
        right_keys = set()

    left_df_input = dataframe_state_left.df_input_copy
    right_df_input = dataframe_state_right.df_input_copy
    df_output = dataframe_state_left.df_output

    used_features = set()

    left_suffix = suffixes[0]
    right_suffix = suffixes[1]

    # Get columns of left, right, and output dataframes
    left_columns = left_df_input.columns
    right_columns = right_df_input.columns
    output_columns = df_output.columns.difference(['_merge'])

    # Identify common keys and columns
    common_keys = left_keys.intersection(right_keys)
    common_columns = left_columns.intersection(right_columns).difference(common_keys)

    # Convert output dataframe to dictionary of records
    records = df_output.to_dict('index')

    generated_entities = []
    used_entities = []
    index_col_to_input_entities = {}

    # Iterate over each record in the output dataframe
    for index, row in records.items():

        # Calculate hash values for the left and right rows
        left_hash_row = sum(
            [hash(str(row[e + left_suffix])) if e in common_columns else hash(str(row[e])) for e in
             left_df_input.columns])
        right_hash_row = sum(
            [hash(str(row[e + right_suffix])) if e in common_columns else hash(str(row[e])) for e in
             right_df_input.columns])

        # Iterate over output columns
        for col_name in output_columns:

            output_value = row[col_name]

            # Create generated entity for the output value
            generated_entity = tracker.global_state.create_entity(value=output_value, feature_name=col_name,
                                                                  index=index,
                                                                  instance=tracker.global_state.operation_number)
            generated_entities.append(generated_entity['id'])
            index_col_to_input_entities[(index, col_name)] = generated_entity

            # Process left-only or both cases
            if row['_merge'] == 'left_only' or row['_merge'] == 'both':

                set_of_indexes = dataframe_state_left.hash_rows_to_indexes.get(left_hash_row, set())

                for left_index in set_of_indexes:

                    if col_name in left_columns or col_name.removesuffix(
                            left_suffix) in common_columns or col_name in common_keys:

                        # Get and remove the used entity from the left dataframe state
                        used_entity = dataframe_state_left.index_col_to_input_entities.get(
                            (left_index, col_name.removesuffix(left_suffix)), None)

                        if used_entity is None:
                            continue

                        used_features.add(col_name)
                        used_entities.append(used_entity['id'])

                        # Create derivation relation between used and generated entities
                        tracker.global_state.create_derivation(used_ent=used_entity['id'],
                                                               gen_ent=generated_entity['id'])

            # Process right-only or both cases
            if row['_merge'] == 'right_only' or row['_merge'] == 'both':
                set_of_indexes = dataframe_state_right.hash_rows_to_indexes.get(right_hash_row, set())

                for right_index in set_of_indexes:
                    if col_name in right_columns or col_name.removesuffix(
                            right_suffix) in common_columns or col_name in common_keys:

                        # Get and remove the used entity from the right dataframe state
                        used_entity = dataframe_state_right.index_col_to_input_entities.get(
                            (right_index, col_name.removesuffix(right_suffix)), None)

                        if used_entity is None:
                            continue

                        used_features.add(col_name)
                        used_entities.append(used_entity['id'])

                        # Create derivation relation between used and generated entities
                        tracker.global_state.create_derivation(used_ent=used_entity['id'],
                                                               gen_ent=generated_entity['id'])

    # Collect invalidated entities
    invalidated = []

    for index in dataframe_state_left.index_col_to_input_entities:
        invalidated.append(dataframe_state_left.index_col_to_input_entities[index]['id'])

    for index in dataframe_state_right.index_col_to_input_entities:
        invalidated.append(dataframe_state_right.index_col_to_input_entities[index]['id'])

    invalidated.extend(used_entities)

    # Create activity and relation in the global state
    act_id = tracker.global_state.create_activity(function_name=function_name, used_features=list(used_features),
                                                  description=tracker.global_state.description,
                                                  code=tracker.global_state.code,
                                                  code_line=tracker.global_state.code_line,
                                                  tracker_id=dataframe_state_left.tracker_id)

    tracker.global_state.create_relation(act_id=act_id, generated=generated_entities, used=used_entities,
                                         invalidated=invalidated,
                                         same=False)

    # Update the index_col_to_input_entities for the left and right dataframe states
    dataframe_state_left.index_col_to_input_entities = index_col_to_input_entities
    dataframe_state_right.index_col_to_input_entities = {}

    # Remove the '_merge' column from the output dataframe if the merge feature is not required
    if not _merge_feature:
        del df_output['_merge']
