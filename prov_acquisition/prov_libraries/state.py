from typing import Dict, List
import uuid
import pandas as pd
from tabulate import tabulate

from misc.logger import CustomLogger
import prov_acquisition.constants as constants

from misc.utils import convert_to_int_no_decimal


class GlobalState:
    """
    Represents the global state of the provenance tracker.

    """

    def __init__(self) -> None:
        """
        Initializes the GlobalState object.

        :return: None
        """

        self.__logger = CustomLogger('ProvenanceTracker')

        self.dataframes_to_state = {}

        self.last_activities = []
        self.current_activities = []
        self.current_entities = []
        self.current_derivations = []
        self.current_relations = []

        self.operation_number = 0
        self.description = None
        self.code_line = None
        self.code = None
        self.function = None

    def update_basic_property(self, description: str, code_line: str, code: str, function: str) -> None:
        """
        Updates the basic properties of the provenance tracker.

        :param description: The description of the current operation.
        :param code_line: The code line of the current operation.
        :param code: The code of the current operation.
        :param function: The function name of the current operation.
        :return: None
        """

        self.description = description
        self.code_line = code_line
        self.code = code
        self.function = function

    def add_dataframes(self, tracker_id: str, state) -> None:
        """
        Adds the DataFrameState object to the dataframes_to_state dictionary.

        :param tracker_id: The tracker ID.
        :param state: The DataFrameState object.
        :return: None
        """

        self.dataframes_to_state[tracker_id] = state

    def create_activity(self, function_name: str, used_features: List[any] = None, description: str = None, other_attributes: Dict[str, any] = None,
                        generated_features: List[any] = None, generated_records: List[any] = None, deleted_used_features: List[any] = None,
                        deleted_records: List[any] = None, code: str = None,
                        code_line: str = None, tracker_id: str = None) -> str:
        """
        Create a provenance activity and add it to the current activities list.
        Return the ID of the new provenance activity.

        :param function_name: The name of the function.
        :param used_features: The list of used features.
        :param description: The description of the activity.
        :param other_attributes: Other attributes of the activity.
        :param generated_features: The list of generated features.
        :param generated_records: The list of generated records.
        :param deleted_used_features: The list of deleted used features.
        :param deleted_records: The list of deleted records.
        :param code: The code of the activity.
        :param code_line: The code line of the activity.
        :param tracker_id: The tracker ID.
        :return: The ID of the new provenance activity.
        """

        act_id = constants.NAMESPACE_ACTIVITY + str(uuid.uuid4())

        attributes = {
            'id': act_id,
            'function_name': function_name,
            'used_features': used_features or [],
            'description': description,
            'generated_features': generated_features,
            'generated_records': generated_records,
            'deleted_used_features': deleted_used_features,
            'deleted_records': deleted_records,
            'code': code,
            'code_line': code_line,
            'tracker_id': constants.NAMESPACE_TRACKER + tracker_id if tracker_id is not None else None,
            'operation_number': str(self.operation_number)
        }

        if other_attributes is not None:
            attributes.update(other_attributes)

        self.current_activities.append(attributes)

        return act_id

    def create_entity(self, value, feature_name: str, index: int, instance: str) -> Dict[str, any]:
        """
        Create a provenance entity.
        Return a dictionary with the ID and the record ID of the entity.

        :param value: The value of the entity.
        :param feature_name: The feature name of the entity.
        :param index: The index of the entity.
        :param instance: The instance of the entity.
        :return: A dictionary with the ID and the record ID of the entity.
        """

        entity = {
            'id': constants.NAMESPACE_ENTITY + str(uuid.uuid4()),
            'value': value,
            'type': type(value).__name__,
            'feature_name': feature_name,
            'index': index,
            'instance': [instance]
        }

        self.current_entities.append(entity)
        return entity

    def create_relation(self, act_id: str, generated: List[any] = None, used: List[any] = None, invalidated: List[any] = None, same: bool = False) -> None:
        """
        Create a provenance relation and add it to the current relations list.

        :param act_id: The ID of the activity.
        :param generated: The list of generated entities.
        :param used: The list of used entities.
        :param invalidated: The list of invalidated entities.
        :param same: A boolean indicating whether the generated and used entities are the same.
        :return: None
        """
       
        generated = generated or []
        used = used or []
        invalidated = invalidated or []

        if same:
            invalidated = []

        self.current_relations.append(
            (generated, used, invalidated, same, act_id))

    def create_derivation(self, used_ent: str, gen_ent: str, add: bool = True) -> Dict[str, any]:
        """
        Create a provenance derivation.

        :param used_ent: The ID of the used entity.
        :param gen_ent: The ID of the generated entity.
        :param add: A boolean indicating whether to add the derivation to the current derivations list.
        :return: A dictionary representing the derivation.
        """
        derivation = {
            'gen': gen_ent,
            'used': used_ent
        }

        if add:
            self.current_derivations.append(derivation)

        return derivation

    def init_prov_entities(self, tracker_id: str) -> None:
        """
        Initializes the provenance entities of the input DataFrame.

        - self.index_col_to_input_entities: Create a dictionary with keys as (index, feature) tuples to enable direct access to the newly created entities.
        - self.hash_rows_to_indexes: Create an inverse dictionary with keys as row hashes to enable access to the corresponding index list.

        :param tracker_id: The tracker ID.
        :return: None
        """

        dataframe_state = self.dataframes_to_state[tracker_id]

        dataframe_state.index_col_to_input_entities = {}

        records = dataframe_state.df_input.to_dict('index')
        for index, row in records.items():
            for col in dataframe_state.df_input.columns:
                dataframe_state.index_col_to_input_entities[(index, col)] = self.create_entity(value=row[col],
                                                                                               feature_name=col,
                                                                                               index=index,
                                                                                               instance=self.operation_number)
            hash_row = sum([hash(str(e)) for e in row.values()])
            set_of_indexes = dataframe_state.hash_rows_to_indexes.get(
                hash_row, set())
            set_of_indexes.add(index)
            dataframe_state.hash_rows_to_indexes[hash_row] = set_of_indexes

        dataframe_state.update_hash_df_input()

    def init_all_prov_entities(self) -> None:
        """
        Initializes the provenance entities for all tracked DataFrames.

        :return: None
        """

        for tracker_id in self.dataframes_to_state:
            self.init_prov_entities(tracker_id=tracker_id)

    def print_current_activities_info(self) -> None:
        """
        Prints the information of the current activities.

        :return: None
        """

        for i, activity in enumerate(self.current_activities):
            table_info = [[f'Activity #{i + 1}', '']]
            for key in activity:
                table_info.append([key, activity[key]])
            activity_info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
            self.__logger.info(f'\n{activity_info}')


class DataFrameState():

    def __init__(self, tracker_id: str) -> None:

        """
        Initializes a DataFrameState object.

        :param tracker_id: The tracker ID.
        """

        self.tracker_id = tracker_id

        self.df_input = None
        self.df_input_copy = None
        self.output = None

        self.index_col_to_input_entities = {}

        self.hash_df_input = None
        self.hash_df_output = None
        self.hash_df_output_common_index = None
        self.hash_rows_to_indexes = {}

    def update_hash_df_output(self) -> any:
        """
        Updates the hash value for the DataFrame output.

        :return: The updated hash value.
        """
        self.hash_df_output = self.df_output.copy().apply(
            lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]), axis=0)
        return self.hash_df_output

    def update_hash_df_input(self) -> any:
        """
        Updates the hash value for the DataFrame input.

        :return: The updated hash value.
        """
        self.hash_df_input = self.df_input.copy().apply(
            lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]), axis=0)
        return self.hash_df_input

    def update_hash_df_output_common_index(self) -> pd.DataFrame:
        """
        Updates the hash value for the DataFrame output with common index.

        :return: The updated hash value.
        """

        int_index = self.df_input_copy.index.intersection(self.df_output.index)
        hash_df_output_common_index = self.df_output.copy().loc[int_index]

        # Calculate hash for non-null columns
        hash_non_null = hash_df_output_common_index.dropna(axis=1, how='all').apply(
            lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]), axis=0)

        # Calculate hash for null columns
        null_columns = hash_df_output_common_index.columns[hash_df_output_common_index.isnull(
        ).all()]
        hash_null = hash_df_output_common_index[null_columns].apply(
            lambda x: sum([hash(str(uuid.uuid4())) for _ in x.values]), axis=0)

        # Concatenate the hash values for non-null and null columns
        self.hash_df_output_common_index = pd.concat(
            [hash_non_null, hash_null])

        return self.hash_df_output_common_index

    def update_hash_row(self) -> Dict[int, any]:
        """
        Updates the hash value for each row in the DataFrame input.
        This method calculates the hash value for each row in the input DataFrame and stores the corresponding indexes 
        in a dictionary.

        :return: The updated hash values with corresponding indexes.
        """
            
        records = self.df_input.to_dict('index')
        for index, row in records.items():
            hash_row = sum([hash(str(e)) for e in row.values()])
            set_of_indexes = self.hash_rows_to_indexes.get(hash_row, set())
            set_of_indexes.add(index)
            self.hash_rows_to_indexes[hash_row] = set_of_indexes

        return self.hash_rows_to_indexes
