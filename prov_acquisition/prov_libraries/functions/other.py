from misc.decorators import suppress_tracking, timing
from prov_acquisition.prov_libraries.state import DataFrameState


@suppress_tracking
@timing
def get_prov_no_change(tracker, dataframe_state: DataFrameState) -> dict:
    """
    Captures the provenance related to a no change operation.

    :param dataframe_state: Input and output DataFrame state.
    :return: None
    """

    function_name = "No Change"

    tracker.global_state.create_activity(function_name, used_features=[],
                                                      description=tracker.global_state.description,
                                                      code=tracker.global_state.code,
                                                      code_line=tracker.global_state.code_line,
                                                      generated_features=[],
                                                      tracker_id=dataframe_state.tracker_id)