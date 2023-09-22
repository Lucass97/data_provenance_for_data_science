from misc.decorators import suppress_tracking, timing
from misc.logger import CustomLogger
from prov_acquisition import constants
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing(log_file=constants.FUNCTION_EXECUTION_TIMES)
def get_prov_space_transformation(tracker, dataframe_state: DataFrameState,
                                  feature_mapping: dict = {}) -> None:
    """
    Captures the provenance related to a change in the dataframe's dimensionality.
    This function uses indexes and can only be used if the operation for capturing provenance does not involve reindexing.

    Types of operations captured by this method:
        - Feature Selection: One or more features are removed.
        - Feature Augmentation: One or more features are added.
        - Instance Drop: One or more records are removed.
        - Instance Generation: One or more records are added.
        - Dimensionality Reduction: Features and records are added/removed. The overall number of removed features and records is greater than those added.
        - Space Augmentation: Features and records are added/removed. The overall number of added features and records is greater than those removed.
        - Space Transformation: Features and records are added/removed. In this case, there can be a reduction in dimensionality for one axis and a space augmentation for the other.
    
    :param tracker: Provenance Tracker
    :param dataframe_state: Input and output DataFrame state.
    :param feature_mapping: Mapping between the features of the output DataFrame and the input DataFrame.
    :return: None
    """

    function_name1 = "Feature Selection"
    function_name2 = "Feature Augmentation"

    function_name3 = "Instance Drop"
    function_name4 = "Instance Generation"

    function_name5 = "Dimensionality Reduction"
    function_name6 = "Space Augmentation"
    function_name7 = "Space Transformation"

    logger = CustomLogger('ProvenanceTracker')
    global_state = tracker.global_state

    df_input = dataframe_state.df_input_copy
    df_output = dataframe_state.df_output

    dropped_rows = df_input.index.difference(df_output.index)
    dropped_cols = df_input.columns.difference(df_output.columns)
    augs_rows = df_output.index.difference(df_input.index)
    augs_cols = df_output.columns.difference(df_input.columns)
    int_rows = df_output.index.intersection(df_input.index)

    used_entities = []
    generated_entities = []
    used_cols = set()

    logger.info(f' Dropped cols: {dropped_cols}')
    logger.info(f' Dropped rows: {dropped_rows}')
    logger.info(f' Generated rows: {augs_rows}')
    logger.info(f' Generated cols: {augs_cols}')

    output_values = df_output.to_numpy()

    # Determine the type of function
    function_name = function_name7
    if len(dropped_cols) > 0 and len(augs_cols) == 0 and len(dropped_rows) == 0 and len(augs_rows) == 0:
        function_name = function_name1
    if (len(dropped_cols) < len(augs_cols)) and len(dropped_rows) < len(augs_cols):
        function_name = function_name6
    if len(dropped_cols) == 0 and len(augs_cols) > 0 and len(dropped_rows) == 0 and len(augs_rows) == 0:
        function_name = function_name2
    if len(dropped_cols) == 0 and len(augs_cols) == 0 and len(dropped_rows) > 0 and len(augs_rows) == 0:
        function_name = function_name3
    if len(dropped_cols) == 0 and len(augs_cols) == 0 and len(dropped_rows) == 0 and len(augs_rows) > 0:
        function_name = function_name4
    if (len(dropped_cols) > len(augs_cols)) and len(dropped_rows) > len(augs_cols):
        function_name = function_name5

    # Iterate over the removed rows to find values deleted due to an Instance Drop operation
    print(feature_mapping)
    for index in dropped_rows:
        for _, col_name in feature_mapping.items():
            used_entity = dataframe_state.index_col_to_input_entities.pop((index, col_name), None)
            if used_entity:
                used_cols.add(col_name)
                used_entities.append(used_entity['id'])

    # Iterate over the added rows to find values added due to an Instance Generation operation
    for index in augs_rows:
        i = df_output.index.get_loc(index)
        for col_name, _ in feature_mapping.items():
            col = df_output.columns.get_loc(col_name)
            output_value = output_values[i, col]
            generated_entity = global_state.create_entity(value=output_value, feature_name=col_name,
                                                          index=index,
                                                          instance=global_state.operation_number)
            dataframe_state.index_col_to_input_entities[(index, col_name)] = generated_entity
            generated_entities.append(generated_entity['id'])

    # Iterate over the remaining rows to find values added/removed due to feature removal/addition
    for index in int_rows:

        i = df_output.index.get_loc(index)

        for col_name in dropped_cols:

            if col_name in set(feature_mapping.values()):
                continue

            used_entity = dataframe_state.index_col_to_input_entities.pop((index, col_name), None)
            if used_entity:
                used_cols.add(col_name)
                used_entities.append(used_entity['id'])

        for col_name in augs_cols:

            if col_name in feature_mapping:
                continue

            col = df_output.columns.get_loc(col_name)
            output_value = output_values[i, col]
            generated_entity = global_state.create_entity(value=output_value, feature_name=col_name,
                                                          index=index,
                                                          instance=global_state.operation_number)
            dataframe_state.index_col_to_input_entities[(index, col_name)] = generated_entity
            generated_entities.append(generated_entity['id'])

    if len(generated_entities) > 0 or len(used_entities) > 0:
        act_id = global_state.create_activity(function_name, used_features=list(used_cols),
                                              description=global_state.description,
                                              code=global_state.code,
                                              code_line=global_state.code_line,
                                              generated_records=len(augs_rows) > 0,
                                              generated_features=list(augs_cols),
                                              deleted_records=len(dropped_rows) > 0,
                                              deleted_used_features=list(dropped_cols),
                                              tracker_id=dataframe_state.tracker_id)

        global_state.create_relation(act_id=act_id, generated=generated_entities, used=used_entities,
                                     invalidated=None, same=True)