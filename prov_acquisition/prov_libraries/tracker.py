import inspect
import logging
import pandas as pd
import uuid
from typing import Union, List

from misc.decorators import timing, suppress_tracking
from misc.logger import CustomLogger
from misc.utils import extract_used_features
from prov_acquisition.prov_libraries.functions.aggregate import get_prov_from_aggregate
from prov_acquisition.prov_libraries.functions.data_combination import get_prov_join
from prov_acquisition.prov_libraries.functions.data_transformation import get_prov_value_change
from prov_acquisition.prov_libraries.functions.features_manipulation import get_prov_feature_rename
from prov_acquisition.prov_libraries.functions.other import get_prov_no_change
from prov_acquisition.prov_libraries.functions.space_transformation import get_prov_space_transformation
from prov_acquisition.prov_libraries.state import GlobalState, DataFrameState
from prov_acquisition.prov_libraries.tracked_dataframe import TrackedDataFrameGroupByMeta, TrackedDataFrameMeta
from prov_acquisition.repository.neo4j import Neo4jFactory, Neo4jConnector


class ProvenanceTracker:
    """
    Class that tracks changes in dataframes and traces provenance.
    """

    def __init__(self, save_on_neo4j: bool = True) -> None:

        self.logger = CustomLogger('ProvenanceTracker')
        self.logger.set_level(logging.DEBUG)

        self.__dataframe_tracking = True
        self.enable_dataframe_warning_msg = True

        self.save_on_neo4j = save_on_neo4j

        self.global_state = GlobalState()

        self.neo4j = Neo4jFactory.create_neo4j_queries(uri="bolt://localhost",
                                                       user="neo4j",
                                                       pwd="adminadmin")
        if self.__save_on_neo4j:
            self.neo4j.delete_all()

    @property
    def save_on_neo4j(self) -> bool:
        return self.__save_on_neo4j
    
    @save_on_neo4j.setter
    def save_on_neo4j(self, value: bool) -> None:
        if value:
            self.logger.warning(f'Enabled saving provenance to Neo4j!')
        else:
            self.logger.warning(f'Disabled saving provenance to Neo4j!')
        self.__save_on_neo4j = value

    @property
    def dataframe_tracking(self) -> bool:
        return self.__dataframe_tracking

    @dataframe_tracking.setter
    def dataframe_tracking(self, value: bool) -> None:
        if self.enable_dataframe_warning_msg:
            if value:
                self.logger.warning(f' Wrapper dataframe provenance tracker was enabled!')
            else:
                self.logger.warning(f' Wrapper dataframe provenance tracker was disabled! Please use track_provenance'
                                    f' method for tracking provenance.')
        self.__dataframe_tracking = value

    @suppress_tracking
    def _wrapper_track_provenance(self, f, tracker_id: str):
        """
        Wrapper function for tracking provenance.

        This function acts as a wrapper for tracking provenance when a method of the TrackedDataFrame class is invoked.
        It triggers the wrap function to capture the provenance.

        :param f: The method or function to be wrapped for provenance tracking.
        :param tracker_id: The unique identifier of the tracker associated with the DataFrame.
        """

        def wrap(*args, **kwargs):

            dataframe_state = self.global_state.dataframes_to_state[tracker_id]

            # print('args', args)
            # print("kwargs", kwargs)

            if callable(f):

                self.logger.info(f' Invoking {f.__name__} function')
                calling_function = inspect.stack()[1].function
                code = inspect.stack()[1].code_context[0].strip(' ').strip('\n')

                used_features = None
                if f.__name__ == '__item__':
                    used_features = extract_used_features(code, dataframe_state.df_input.columns)
                    column_to_add = args[1]

                if f.__name__ == 'merge':
                    kwargs['indicator'] = True

                result = f(*args, **kwargs)

                # print({f.__name__}, '\n', result, args, kwargs)

                self.logger.info(f' The function {f.__name__} was called by {calling_function} function.')

                if hasattr(dataframe_state.df_input, calling_function):
                    self.logger.info(
                        f' The function {f.__name__} was called by {calling_function} function. Skipping data provenance phase.')
                    return result

                dataframe_state.df_output = result

                # Preparazione dei metadati per la cattura della provenance. Se si traccia il DataframeGroupby allora non si aggiorna.
                if f.__name__ != "_wrap_agged_manager":
                    self.global_state.update_basic_property(description=f.__name__,
                                                            code=code,
                                                            code_line=inspect.stack()[1].lineno,
                                                            function=f.__name__)

                if isinstance(result, pd.core.groupby.generic.DataFrameGroupBy):
                    return self.create_tracked_dataframe_groupby(result.obj, tracker_id=tracker_id)

                # Nel caso il risultato sia un dataframe allora si esegue la cattura della provenance.
                if isinstance(result, pd.DataFrame) or result is None:

                    # In caso di operazione inplace.
                    if result is None:
                        dataframe_state.df_output = dataframe_state.df_input
                        dataframe_state.df_input = dataframe_state.df_input_copy

                    if f.__name__ == '__setitem__':
                        dataframe_state.df_output = args[0]

                    if not self.__dataframe_tracking:
                        self.logger.warning(
                            f' Wrapper dataframe provenance is disabled! Data provenance will not be caught.')
                        self.__prepare_for_next_operation(dataframe_state=dataframe_state,
                                                          update_df_input=False, save=False)
                        if result is None:
                            return result
                        else:
                            return self.create_tracked_dataframe(result, tracker_id=tracker_id)

                    self.global_state.operation_number += 1

                    dataframe_state.update_hash_df_output()
                    dataframe_state.update_hash_df_output_common_index()

                    if f.__name__ == '_wrap_agged_manager':
                        get_prov_from_aggregate(self, dataframe_state=dataframe_state, remaining_features=args[1].items)

                    elif self.global_state.function == 'merge':

                        right_df_input = kwargs['right'] if 'right' in kwargs else args[1] if len(args) >= 1 else None

                        on = kwargs['on'] if 'on' in kwargs else args[3] if len(args) >= 4 else None
                        left_on = kwargs['left_on'] if 'left_on' in kwargs else args[4] if len(args) >= 5 else None
                        right_on = kwargs['right_on'] if 'right_on' in kwargs else args[5] if len(args) >= 6 else None
                        suffixes = kwargs['suffixes'] if 'suffixes' in kwargs else '_x', '_y'

                        if on is not None:
                            left_on = on
                            right_on = on

                        if kwargs['how'] == 'cross':
                            left_on = set()
                            right_on = set()

                        self.global_state.operation_number -= 1

                        dataframe_state_right = self.global_state.dataframes_to_state[right_df_input.tracker_id]
                        dataframe_state.update_hash_row()
                        dataframe_state_right.update_hash_row()
                        self.global_state.operation_number += 1

                        get_prov_join(self, dataframe_state_left=dataframe_state,
                                      dataframe_state_right=dataframe_state_right, how=kwargs['how'],
                                      left_keys=set(left_on),
                                      right_keys=set(right_on), suffixes=suffixes)
                    else:
                        feature_mapping = get_prov_feature_rename(self, dataframe_state=dataframe_state)

                        get_prov_space_transformation(self, dataframe_state=dataframe_state,
                                                      feature_mapping=feature_mapping)

                        get_prov_value_change(self, dataframe_state=dataframe_state, extra_used_features=used_features)

                    if not self.global_state.current_activities:
                        get_prov_no_change(self, dataframe_state=dataframe_state)

                    self.global_state.print_current_activities_info()

                    self.__prepare_for_next_operation(dataframe_state=dataframe_state,
                                                      update_df_input=True)

                if result is None:
                    return result
                elif isinstance(result, pd.DataFrame):
                    return self.create_tracked_dataframe(result, tracker_id=tracker_id)
                else:
                    return result

        return wrap

    @suppress_tracking
    @timing
    def check_equals_dataframe(self, feature_mapping: dict) -> bool:
        """
        Check if the df_input and df_output dataframes are equal.

        This method compares the df_input and df_output dataframes to determine if they are identical
        after applying a feature mapping.

        :param feature_mapping: A dictionary mapping features (columns) from df_output to df_input.
        :return: True if the dataframes are equal; False otherwise.
        """

        function_name = "Check Equals Dataframe"

        result = False
        hash_df_output = self.hash_df_output.copy()
        hash_df_output.rename(feature_mapping)
        if self._df_output is not None and self.hash_df_output is not None:
            result = self.hash_df_input.equals(hash_df_output)

        if result:
            self.logger.info(f' {function_name}: dataframe are equals!')

        return result

    def __prepare_for_next_operation(self, dataframe_state: DataFrameState, update_df_input: bool = True,
                                     save: bool = True) -> None:
        """
        Prepare for the next operation.

        This method prepares the environment and data for the next operation.

        :param dataframe_state: The state of the DataFrame.
        :param update_df_input: Whether to update the input DataFrame.
        :param save: Whether to save data to Neo4j.
        :return: None
        """
        self.global_state.description = None

        dataframe_state.df_input = dataframe_state.df_output

        if update_df_input:
            dataframe_state.df_input_copy = dataframe_state.df_input.copy()
            dataframe_state.hash_df_input = dataframe_state.hash_df_output

        dataframe_state.hash_df_output = None

        

        # Save to neo4j
        if self.__save_on_neo4j:

            session = Neo4jConnector().create_session()

            # Create constraints in Neo4j
            self.neo4j.create_constraint(session=session)

            # Add activities, entities, derivations, and relations to Neo4j
            self.neo4j.add_activities(self.global_state.current_activities, session)
            self.neo4j.add_entities(self.global_state.current_entities)
            self.neo4j.add_derivations(self.global_state.current_derivations)
            self.neo4j.add_relations(self.global_state.current_relations)

            if self.global_state.last_activities:
                # Determine next operations based on current and last activities
                next_operations = [{'act_in_id': a_in['id'], 'act_out_id': a_out['id']}
                                   for a_out in self.global_state.current_activities
                                   for a_in in self.global_state.last_activities
                                   if a_out['tracker_id'] == a_in['tracker_id']]
                self.neo4j.add_next_operations(next_operations, session)

            # Update the last activities in the global state
            self.global_state.last_activities = self.global_state.current_activities.copy()

            # Free memory by clearing lists
            del self.global_state.current_activities[:]
            del self.global_state.current_entities[:]
            del self.global_state.current_derivations[:]
            del self.global_state.current_relations[:]

    def create_tracked_dataframe(self, df: pd.DataFrame, tracker_id: str):
        """
        Create a tracked dataframe.

        This method creates a new tracked dataframe by inheriting from the `pd.DataFrame` class and
        applying the tracking functionality defined in the `TrackedDataFrameMeta` metaclass. Each
        tracked dataframe is associated with a specific tracker instance and a unique tracker ID.

        :param df: The original DataFrame that you want to track.
        :param tracker_id: A unique identifier for the tracker instance associated with this DataFrame.
        :return: A new tracked DataFrame.
        """

        class DataFrameTracked(pd.DataFrame, metaclass=TrackedDataFrameMeta, tracker=self, tracker_id=tracker_id):
            pass

        return DataFrameTracked(df)

    def create_tracked_dataframe_groupby(self, df_groupby: pd.core.groupby.generic.DataFrameGroupBy, tracker_id: str):
        """
        TODO
        :param df_groupby:
        :param tracker_id:
        :return:
        """

        class DataFrameGroupByTracked(pd.core.groupby.generic.DataFrameGroupBy, metaclass=TrackedDataFrameGroupByMeta,
                                      tracker=self,
                                      tracker_id=tracker_id):
            pass

        return DataFrameGroupByTracked(df_groupby, [df_groupby['A'], df_groupby['B']])

    def subscribe(self, df: Union[pd.DataFrame, List[pd.DataFrame]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Subscribe to a DataFrame or a list of DataFrames.

        This method allows subscribing to either a single DataFrame or a list of DataFrames for tracking purposes.

        :param df: A single DataFrame or a list of DataFrames to subscribe to.
        :return: If a single DataFrame is passed, returns the tracked DataFrame. If a list is passed,
                 returns a list of tracked DataFrames.
        """

        # In the case where only one dataframe is passed
        if isinstance(df, pd.DataFrame):
            tracker_id = str(uuid.uuid4())
            dataframe_state = DataFrameState(tracker_id=tracker_id)

            df_tracked = self.create_tracked_dataframe(df=df, tracker_id=tracker_id)

            dataframe_state.df_input = df_tracked
            dataframe_state.df_input_copy = df_tracked.copy()

            self.global_state.add_dataframes(df_tracked.tracker_id, dataframe_state)
            self.global_state.init_prov_entities(tracker_id=df_tracked.tracker_id)

            return df_tracked

        # In the case where a list of dataframes is passed
        elif isinstance(df, list):
            tracked_dfs = []
            for single_df in df:
                tracker_id = str(uuid.uuid4())
                dataframe_state = DataFrameState(tracker_id=tracker_id)

                df_tracked = self.create_tracked_dataframe(df=single_df, tracker_id=tracker_id)

                dataframe_state.df_input = df_tracked
                dataframe_state.df_input_copy = df_tracked.copy()

                self.global_state.add_dataframes(df_tracked.tracker_id, dataframe_state)
                self.global_state.init_prov_entities(tracker_id=df_tracked.tracker_id)

                tracked_dfs.append(df_tracked)

            return tracked_dfs

        else:
            raise ValueError("Invalid input format. Expected a single DataFrame or a list of DataFrames.")
