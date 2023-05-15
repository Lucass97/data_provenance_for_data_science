import inspect
import logging
import uuid
from typing import Iterable, Tuple

import pandas as pd

from misc.utils import timing, suppress_tracking, extract_used_features, keys_mapping, \
    convert_to_int_no_decimal
from prov_acquisition.repository.neo4j_query import Neo4jConnection


class ProvenanceTracker:
    """
    Classe che tracka i cambiamenti nei dataframe e traccia la provenance.
    """

    # Constants:
    NAMESPACE_FUNC = 'activity:'
    NAMESPACE_ENTITY = 'entity:'
    INPUT = 'input'
    OUTPUT = 'output'
    LIST_REL_SIZE = 25000
    CHUNK_SIZE = 300000
    CHUNK_INDEX_SIZE = 1000

    # PROV-N objects
    ENTITY = 'prov:entity'
    GENERATED_ENTITY = 'prov:generatedEntity'
    USED_ENTITY = 'prov:usedEntity'
    ACTIVITY = 'prov:activity'

    # PROV-N relations
    GENERATION = 'gen'
    USE = 'used'
    DERIVATION = 'wasDerivedFrom'
    INVALIDATION = 'wasInvalidatedBy'

    DEFAULT_PATH = 'prov_results/'

    SEPARATOR = '\x1e'
    SEPARATOR = '^^'

    _dataframe_tracking = True
    enable_dataframe_warning_msg = True

    df_input_copy = None
    _df_output = None
    df_input = None
    hash_df_input = None
    hash_df_output = None
    hash_df_output_common_index = None

    def __init__(self, df: pd.DataFrame):

        logging.debug(f"{self.__class__} Init")

        # pandarallel.initialize(progress_bar=True)

        self.df_input = self.create_tracked_dataframe(df=df)
        self.df_input_copy = self.df_input.copy()

        # Initializa shapes

        self.m_input, self.n_shape = self.df_input.shape
        self.m_output, self.n_output = 0, 0

        # Selected df

        self.index_col_to_input_entities = {}
        self.index_col_to_output_entities = {}
        self.hash_rows_to_indexes = {}
        self.reindexing = False

        # Second df for join operation

        self.index_col_to_input_entities_2 = {}
        self.hash_rows_to_indexes_2 = {}

        self.last_activities = []
        self.current_activities = []
        self.current_entities = []
        self.current_derivations = []
        self.current_relations = []
        self.operation_number = 0

        # Current activity attributes

        self.description = None
        self.code_line = None
        self.code = None
        self.function = None

        self.init_prov_entities()

        # self.saver = Saver()

        self.neo4j = Neo4jConnection(uri="bolt://localhost",
                                     user="neo4j",
                                     pwd="adminadmin")

        self.neo4j.delete_all()

    @property
    def df_output(self):
        return self._df_output

    @df_output.setter
    def df_output(self, value: pd.DataFrame):
        logging.warning(f' Can\'t assign a new value to df_output')

    @property
    def dataframe_tracking(self):
        return self._dataframe_tracking

    @dataframe_tracking.setter
    def dataframe_tracking(self, value: bool):
        if self.enable_dataframe_warning_msg:
            if value:
                logging.warning(f' Wrapper dataframe provenance tracker was enable!')
            else:
                logging.warning(
                    f' Wrapper dataframe provenance tracker was disable! Please use track_provenance method for tracking provenance.')
        self._dataframe_tracking = value

    def track_provenance(self, f, *args, description: str = None, **kwargs) -> pd.DataFrame:
        """
        Metodo esplicito per la cattura della data provenance.
        TODO
        """
        if callable(f):
            logging.debug(f' Invoking {f.__name__} function with {args} and {kwargs} parameters...')
            self._df_output = f(*args, **kwargs)

            # Preparazione dei metadati per la cattura della provenance.

            if description is None:
                description = f.__name__
            self.description = description
            self.code = inspect.stack()[1].code_context[0].strip(' ').strip('\n')
            self.code_line = inspect.stack()[1].lineno
            self.function = f.__name__
            self.operation_number += 1

            # Cattura la provenance

            self.value_change_provenance(self.df_input, self.df_output)
            self._shape_change_provenance_tracking(df_input=self.df_input, df_output=self._df_output)

            self.prepare_for_next_operation()

        else:
            return self._df_output

    def _wrapper_track_provenance(self, f):
        """
        Funzione wrapper per tracciare la provenance.
        Qualora fosse invocato un metodo della classe TrackedDataframe,
        allora verrà lanciata la funzione wrap per la cattura della provenance.

        """

        def wrap(*args, **kwargs):

            if callable(f):

                logging.info(f' Invoking {f.__name__} function')
                calling_function = inspect.stack()[1].function
                code = inspect.stack()[1].code_context[0].strip(' ').strip('\n')

                used_features = None
                if f.__name__ == '__item__':
                    used_features = extract_used_features(code, self.df_input.columns)

                if f.__name__ == 'merge':
                    kwargs['indicator'] = True

                result = f(*args, **kwargs)

                if hasattr(self.df_input, calling_function):
                    logging.info(
                        f' The function {f.__name__} was called by {calling_function} function. Skipping data provenance phase.')
                    return result

                self._df_output = result

                # Nel caso il risultato sia un dataframe allora si esegue la cattura della provenance.
                if isinstance(result, pd.DataFrame) or result is None:

                    # In caso di operazione inplace.
                    if result is None:
                        self._df_output = self.df_input
                        self.df_input = self.df_input_copy

                    if not self._dataframe_tracking:
                        logging.warning(
                            f' Wrapper dataframe provenance is disable! Data provenance will not be caught.')
                        self.prepare_for_next_operation(update_df_input=False, save=False)
                        return result

                    # Preparazione dei metadati per la cattura della provenance.
                    self.description = f.__name__
                    self.code = code
                    self.code_line = inspect.stack()[1].lineno
                    self.function = f.__name__
                    self.operation_number += 1

                    self.hash_df_output = self._df_output.copy()
                    self.hash_df_output = self.hash_df_output.apply(lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]),
                                                                    axis=0)
                    int_index = self.df_input_copy.index.intersection(self._df_output.index)
                    self.hash_df_output_common_index = self._df_output.copy()
                    self.hash_df_output_common_index = self.hash_df_output_common_index.loc[int_index].apply(
                        lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]),
                        axis=0)

                    # self.generate_dataframe_metadata(df=self.df_input)

                    # Cattura la provenance

                    if self.function == 'merge':

                        right_df_input = kwargs['right'] if 'right' in kwargs else args[1] if len(args) >= 1 else None

                        on = kwargs['on'] if 'on' in kwargs else args[3] if len(args) >= 4 else None
                        left_on = kwargs['left_on'] if 'left_on' in kwargs else args[4] if len(args) >= 5 else None
                        right_on = kwargs['right_on'] if 'right_on' in kwargs else args[5] if len(args) >= 6 else None
                        suffixes = kwargs['suffixes'] if 'suffixes' in kwargs else '_x', '_y'

                        if on:
                            left_on = on
                            right_on = on

                        self.operation_number -= 1
                        self.init_second_df(df_input=right_df_input,
                                            index_col_to_input_entities=self.index_col_to_input_entities_2,
                                            hash_rows_to_indexes=self.hash_rows_to_indexes_2)
                        self.operation_number += 1

                        self.get_prov_join(left_df_input=self.df_input, right_df_input=right_df_input,
                                           df_output=self._df_output, left_keys=set(left_on), right_keys=set(right_on),
                                           suffixes=suffixes)
                    else:
                        feature_mapping = self._get_prov_feature_rename(df_input=self.df_input_copy,
                                                                        df_output=self._df_output)
                        if self.check_equals_dataframe(feature_mapping=feature_mapping):
                            self._get_prov_reindexing(df_output=self._df_output)
                        else:
                            self._get_prov_space_transformation(df_input=self.df_input_copy, df_output=self._df_output,
                                                                feature_mapping=feature_mapping)
                            self.get_prov_value_change(df_input=self.df_input_copy, df_output=self._df_output,
                                                       extra_used_features=used_features)

                    """logging.info(f' operation_number={self.operation_number}')
                    logging.info(f' current_activities={self.current_activities}')
                    logging.info(f' current_entities={self.current_entities}')
                    logging.info(f' current_derivations={self.current_derivations}')
                    logging.info(f' current_relations={self.current_relations}')"""

                    self.prepare_for_next_operation()

                return result

        return wrap

    def _shape_change_provenance_tracking(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> None:
        """
        Cattura la provenance relativa ad un cambiamento della shape del dataframe.
        Richiama le opportune funzioni in base al caso.

        return: None
        """
        function_name = 'shape_change_tracking'
        input_rows, input_cols = df_input.shape
        print(df_output)
        output_rows, output_cols = df_output.shape

        diff_columns = df_output.columns.difference(df_input.columns)

        feature_mapping = self._get_prov_feature_rename(df_input=df_input, df_output=df_output)

        self._get_prov_space_transformation(df_input=df_input, df_output=df_output, feature_mapping=feature_mapping)

        """if output_rows < input_rows:
            logging.info(f" <{function_name}> Instance drop detected: {input_rows} rows -> {output_rows} rows")
            self.get_prov_instance_drop(df_input=df_input, df_output=df_output)
        if output_cols < input_cols:
            logging.info(f" <{function_name}> Space transform detected: {diff_columns}.")
            self.get_prov_dim_reduction(df_input=df_input, df_output=df_output)
        if output_rows > input_rows:
            logging.info(f" <{function_name}> Instance generation detected.")
            self.get_prov_instance_generation(df_input=df_input, df_output=df_output)
        if output_cols > input_cols:
            logging.info(f" <{function_name}> Columns augmentation detected.")
            self.get_prov_columns_aug(df_input=df_input, df_output=df_output)"""

    @suppress_tracking
    @timing
    def get_prov_join(self, left_df_input: pd.DataFrame, right_df_input: pd.DataFrame, df_output: pd.DataFrame,
                      left_keys, right_keys, suffixes: Tuple[str], _merge_feature: bool = False) -> None:
        """
        Cattura la provenance relativa all'operazione di join.
        Problemi noti da risolvere: l'operazione di merge può convertire le colonne di interi in colonne di float
        se queste contengono dei valori nulli. Cambia il tipo, cambia l'hash della riga e dunque
        non si trovano i corrispondenti indici.


        :param left_df_input: primo dataframe in input.
        :param right_df_input: secondo dataframe in input
        :param df_output: dataframe di output
        :param left_keys:
        :param right_keys:
        :param suffixes: Suffissi per le chiavi in comune
        :param _merge_feature: Indica se la feature merge è stata generata precedentemente alla provenance.

        """

        function_name = "Join"

        used_features = set()

        left_suffix = suffixes[0]
        right_suffix = suffixes[1]

        left_columns = left_df_input.columns
        right_columns = right_df_input.columns
        output_columns = df_output.columns.difference(['_merge'])

        common_keys = left_keys.intersection(right_keys)
        common_columns = left_columns.intersection(right_columns).difference(common_keys)

        records = df_output.to_dict('index')

        generated_entities = []
        used_entities = []
        index_col_to_input_entities = {}

        for index, row in records.items():

            left_hash_row = sum(
                [hash(str(row[e + left_suffix])) if e in common_columns else hash(str(row[e])) for e in
                 left_df_input.columns])
            right_hash_row = sum(
                [hash(str(row[e + right_suffix])) if e in common_columns else hash(str(row[e])) for e in
                 right_df_input.columns])

            for col_name in output_columns:

                output_value = row[col_name]
                generated_entity = self.create_entity(value=output_value, feature_name=col_name, index=index,
                                                      instance=self.operation_number)
                generated_entities.append(generated_entity['id'])

                if row['_merge'] == 'left_only' or row['_merge'] == 'both':

                    set_of_indexes = self.hash_rows_to_indexes.get(left_hash_row, set())

                    for left_index in set_of_indexes:

                        if col_name in left_columns or col_name.removesuffix(
                                left_suffix) in common_columns or col_name in common_keys:

                            used_entity = self.index_col_to_input_entities.pop(
                                (left_index, col_name.removesuffix(left_suffix)), None)

                            if used_entity is None:
                                continue

                            used_features.add(col_name)
                            index_col_to_input_entities[(index, col_name)] = generated_entity
                            # used_entities.append(used_entity)
                            self.create_derivation(used_ent=used_entity['id'], gen_ent=generated_entity['id'])

                if row['_merge'] == 'right_only' or row['_merge'] == 'both':
                    set_of_indexes = self.hash_rows_to_indexes_2.get(right_hash_row, set())
                    for right_index in set_of_indexes:
                        if col_name in right_columns or col_name.removesuffix(
                                right_suffix) in common_columns or col_name in common_keys:

                            used_entity = self.index_col_to_input_entities_2.pop(
                                (right_index, col_name.removesuffix(right_suffix)), None)

                            if used_entity is None:
                                continue

                            used_features.add(col_name)
                            index_col_to_input_entities[(index, col_name)] = generated_entity
                            # used_entities.append(used_entity)

                            self.create_derivation(used_ent=used_entity['id'], gen_ent=generated_entity['id'])

        act_id = self.create_activity(function_name=function_name, used_features=list(used_features),
                                      description=self.description, code=self.code, code_line=self.code_line)

        self.create_relation(act_id=act_id, generated=generated_entities, used=used_entities, invalidated=used_entities,
                             same=False)

        self.index_col_to_input_entities = index_col_to_input_entities

        if not _merge_feature:
            del df_output['_merge']

    @suppress_tracking
    @timing
    def _get_prov_union(self, df_inputs, df_output, join: str):
        """
        Cattura la provenance relativa ad un'operazione di union.
        TODO
        """

        pass

    @suppress_tracking
    @timing
    def get_prov_space_transformation_old(self, df_input, df_output) -> None:
        """
        Da cancellare forse
        """

        function_name = "Space Trasformation"

        generated_entities = []
        used_entities = {}
        derivations = {}

        aug_cols = df_output.columns.difference(df_input.columns)
        dropped_cols = df_input.columns.difference(df_output.columns)

        used_cols = set()

        output_values = df_output.to_numpy()

        for index in df_output.index:

            i = df_output.index.get_loc(index)

            for col_name1 in aug_cols:
                col = df_output.columns.get_loc(col_name1)
                output_value = str(output_values[i][col])
                generated_entity = self.create_entity(value=output_value, feature_name=col_name1, index=i,
                                                      instance=self.operation_number)

                self.index_col_to_input_entities[(index, col_name1)] = generated_entity
                generated_entities.append(generated_entity)

                for col_name2 in dropped_cols:

                    used_entity = self.index_col_to_input_entities.get((index, col_name2), None)

                    if used_entity is None:
                        continue

                    input_value = used_entity.split(self.SEPARATOR)[0]

                    used_cols.add(col_name2)

                    list_of_used_entities = used_entities.get(col_name2, list())
                    list_of_used_entities.append(used_entity)
                    used_entities[col_name2] = list_of_used_entities

                    derivation = self.create_derivation(used_ent=used_entity, gen_ent=generated_entity, add=False)

                    list_of_derivations = derivations.get(col_name2, list())
                    list_of_derivations.append(derivation)
                    derivations[col_name2] = list_of_derivations

                    if output_value != input_value:
                        del used_entities[col_name2][:]
                        used_cols.remove(col_name2)
                        break

        a, b = [], []
        for col_name in used_cols:
            a.extend(used_entities[col_name])
            b.extend(derivations[col_name])

        self.current_derivations.extend(b)

        if len(aug_cols) > 0:
            act_id = self.create_activity(function_name=function_name, used_features=list(used_cols),
                                          description=self.description, code=self.code, code_line=self.code_line)
            self.create_relation(act_id=act_id, generated=generated_entities, used=a,
                                 invalidated=a, same=True)

    @suppress_tracking
    @timing
    def get_prov_instance_drop(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> None:
        """
        Cattura la provenance relativa all'operazione di Instance Drop.

        :param df_input: Dataframe in input
        :param df_output: Dataframe in output
        :return: None
        """

        function_name = "Instance Drop"

        invalidated_entities = []
        used_cols = set()

        indexes = df_input.index.difference(df_output.index)

        int_columns = df_output.columns.intersection(df_input.columns)

        for index in indexes:

            for col_name in int_columns:

                entity = self.index_col_to_input_entities.pop((index, col_name), None)

                if entity:
                    invalidated_entities.append(entity['id'])
                    used_cols.add(col_name)

        act_id = self.create_activity(function_name=function_name, used_features=list(used_cols),
                                      description=self.description, code=self.code, code_line=self.code_line,
                                      deleted_records=len(used_cols) > 0)

        self.create_relation(act_id=act_id, generated=None, used=invalidated_entities, invalidated=invalidated_entities,
                             same=True)

    @suppress_tracking
    @timing
    def _get_prov_space_transformation(self, df_input: pd.DataFrame, df_output: pd.DataFrame,
                                       feature_mapping: dict = {}) -> None:
        """
        Cattura la provenance relativa ad un cambiamento della dimensionalità del dataframe.
        Questa funzione fa uso degli indici e, può essere utilizzata solamente
        se l'operazione di cui catturare la provenance non prevede la reindicizzazione.

        Tipi di operazione che cattura questo metodo:
            - Feature Selection: Vengono rimosse una o più feature;
            - Feature Agumentation: Vengono aggiunte una o più feature;
            - Instance Drop: Vengono rimossi uno o più record;
            - Instance Generation: Vengono aggiunti un o più record;
            - Dimensionality Reduction: Vengono aggiunte/rimosse feature e record. Complessivamente il numero di feature e record rimossi è maggiore di quelli aggiunti;
            - Space Augmentation: Vengono aggiunte/rimosse feature e record. Complessivamente il numero di feature e record aggiunti è maggiore di quelli aggiunti;
            - Space Transformation: Vengono aggiunte/rimosse feature e record. In questo caso si può verificare una riduzione di dimnesionalità per un asse mentre per l'altro una space augmentation.


        :param df_input: DataFrame in input
        :param df_output: DataFrame in output
        :param feature_mapping: Corrispondenza tra le feature del df_output con quelle del df_input
        :return: None

        """

        function_name1 = "Feature Selection"
        function_name2 = "Feature Augmentation"

        function_name3 = "Instance Drop"
        function_name4 = "Instance Generation"

        function_name5 = "Dimensionality Reduction"
        function_name6 = "Space Augmentation"
        function_name7 = "Space Transformation"

        dropped_rows = df_input.index.difference(df_output.index)
        dropped_cols = df_input.columns.difference(df_output.columns)
        augs_rows = df_output.index.difference(df_input.index)
        augs_cols = df_output.columns.difference(df_input.columns)
        int_rows = df_output.index.intersection(df_input.index)

        used_entities = []
        generated_entities = []
        used_cols = set()

        logging.info(f' Dropped cols: {dropped_cols}')
        logging.info(f' Dropped rows: {dropped_rows}')
        logging.info(f' Generated rows: {augs_rows}')
        logging.info(f' Generated cols: {augs_cols}')

        output_values = df_output.to_numpy()

        """ Determina il tipo di funzione """

        function_name = function_name7

        if len(dropped_cols) > 0 and len(augs_cols) == 0 and len(dropped_rows) == 0 and len(augs_rows) == 0:
            function_name = function_name1
        if len(dropped_cols) == 0 and len(augs_cols) > 0 and len(dropped_rows) == 0 and len(augs_rows) == 0:
            function_name = function_name2

        if len(dropped_cols) == 0 and len(augs_cols) == 0 and len(dropped_rows) > 0 and len(augs_rows) == 0:
            function_name = function_name3
        if len(dropped_cols) == 0 and len(augs_cols) == 0 and len(dropped_rows) == 0 and len(augs_rows) > 0:
            function_name = function_name4

        if (len(dropped_cols) > len(augs_cols)) and len(dropped_rows) > len(augs_cols):
            function_name = function_name5
        if (len(dropped_cols) < len(augs_cols)) and len(dropped_rows) < len(augs_cols):
            function_name = function_name6

        # Itera sulle righe rimosse per trovare valori cancellati in seguito ad un'operazione di Instance Drop.
        for index in dropped_rows:
            for _, col_name in feature_mapping.items():
                used_entity = self.index_col_to_input_entities.pop((index, col_name), None)
                if used_entity:
                    used_cols.add(col_name)
                    used_entities.append(used_entity['id'])

        # Itera sulle righe aggiunte per trovare valori aggiunti in seguito ad un'operazione di Instance Generation
        for index in augs_rows:
            i = df_output.index.get_loc(index)
            for col_name, _ in feature_mapping.items():
                col = df_output.columns.get_loc(col_name)
                output_value = output_values[i, col]
                generated_entity = self.create_entity(value=output_value, feature_name=col_name, index=index,
                                                      instance=self.operation_number)
                self.index_col_to_input_entities[(index, col_name)] = generated_entity
                generated_entities.append(generated_entity['id'])

        # Itera sulle righe non rimosse per trovare valori cancellati/aggiunti in seguito ad una rimozione/aggiunta di una o più feature.
        for index in int_rows:

            i = df_output.index.get_loc(index)

            for col_name in dropped_cols:

                if col_name in set(feature_mapping.values()):
                    continue

                used_entity = self.index_col_to_input_entities.pop((index, col_name), None)
                if used_entity:
                    used_cols.add(col_name)
                    used_entities.append(used_entity['id'])

            for col_name in augs_cols:

                if col_name in feature_mapping:
                    continue

                col = df_output.columns.get_loc(col_name)
                output_value = output_values[i, col]
                generated_entity = self.create_entity(value=output_value, feature_name=col_name, index=index,
                                                      instance=self.operation_number)
                self.index_col_to_input_entities[(index, col_name)] = generated_entity
                generated_entities.append(generated_entity['id'])

        if len(generated_entities) > 0 or len(used_entities) > 0:
            act_id = self.create_activity(function_name, used_features=list(used_cols), description=self.description,
                                          code=self.code,
                                          code_line=self.code_line,
                                          generated_records=len(augs_rows) > 0,
                                          generated_features=list(augs_cols),
                                          deleted_records=len(dropped_rows) > 0,
                                          deleted_used_features=list(dropped_cols))

            self.create_relation(act_id=act_id, generated=generated_entities, used=used_entities,
                                 invalidated=None, same=True)

    @suppress_tracking
    @timing
    def _get_prov_feature_rename(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> dict:
        """
        Cattura la provenance relativa ad una ridenominazione di una o più feature.

        :param df_input: DataFrame in input
        :param df_output: DataFrame in output
        :return: dict - Corrispondenza tra le feature del df_output con quelle del df_input

        """

        function_name = "Feature Rename"

        int_rows = df_output.index.intersection(df_input.index)

        used_entities = []
        generated_entities = []
        used_features = set()
        generated_features = set()

        output_values = df_output.to_numpy()

        hash_df_output_common_index = self.hash_df_output_common_index.to_dict()
        hash_df_input = self.hash_df_input.to_dict()

        # prende il posto di int_cols
        feature_mapping = keys_mapping(hash_df_output_common_index, hash_df_input)

        for index in int_rows:
            i = df_output.index.get_loc(index)
            for col_name1, col_name2 in feature_mapping.items():

                if col_name1 == col_name2:
                    continue

                col = df_output.columns.get_loc(col_name1)
                output_value = output_values[i, col]

                used_entity = self.index_col_to_input_entities.pop((index, col_name2), None)
                if used_entity:
                    generated_entity = self.create_entity(value=output_value, feature_name=col_name1, index=index,
                                                          instance=self.operation_number)
                    generated_entities.append(generated_entity['id'])
                    used_entities.append(used_entity['id'])
                    generated_features.add(col_name1)
                    used_features.add(col_name2)

                    self.index_col_to_input_entities[(index, col_name1)] = generated_entity
                    self.create_derivation(used_ent=used_entity['id'], gen_ent=generated_entity['id'])

        if len(generated_features) > 0:
            logging.info(f' Feature rename detect: {feature_mapping}')
            act_id = self.create_activity(function_name, used_features=list(used_features),
                                          description=self.description,
                                          code=self.code,
                                          code_line=self.code_line,
                                          generated_features=list(generated_features))

            self.create_relation(act_id=act_id, generated=generated_entities, used=used_entities, invalidated=[],
                                 same=True)

        return feature_mapping

    @suppress_tracking
    @timing
    def get_prov_value_change(self, df_input: pd.DataFrame, df_output: pd.DataFrame,
                              extra_used_features: set = None) -> None:
        """
        Cattura la provenance relativa ad un cambiamento dei valori del dataframe.
        Il tipo di activity generata può essere di due tipi:
        Value Transformation: caso generico.
        Imputation: la colonna del dataframe ha subito una sostituzione di valori nulli.

        :param df_input: Dataframe in input
        :param df_output: Dataframe in output
        :param extra_used_features: Features extra usate in input. Indicano le features aggiuntive che contribuiscono alla generazione delle nuove entità.
        :return: None
        """

        logging.info(f' Check for value change...')

        function_name1 = "Value Transformation"
        function_name2 = "Imputation"

        int_columns = df_output.columns.intersection(df_input.columns)

        generated_entities = []
        used_entities = []
        extra_used_entities = []
        imp_cols = set()
        trans_cols = set()
        values_output = df_output[int_columns].to_numpy()

        if extra_used_features is None:
            used_features = set()

        for i in self._df_output.index:

            index = df_output.index.get_loc(i)

            for col_name in int_columns:

                col = int_columns.get_loc(col_name)
                new_value = values_output[index][col]

                used_entity = self.index_col_to_input_entities.get((index, col_name), None)

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

                    entity = self.create_entity(value=new_value, feature_name=col_name, index=index,
                                                instance=self.operation_number)

                    self.index_col_to_input_entities[(index, col_name)] = entity

                    generated_entities.append(entity['id'])
                    used_entities.append(used_entity['id'])

                    self.create_derivation(used_ent=used_entity['id'], gen_ent=entity['id'])

                    # features extra da aggiungere
                    for used_feature in used_features:
                        used_entity = self.index_col_to_input_entities.get((index, used_feature), None)

                        if used_entity is None:
                            continue

                        extra_used_entities.append(used_entity)
                        self.create_derivation(used_ent=used_entity['id'], gen_ent=entity['id'])

        imp_cols = imp_cols.difference(trans_cols)

        if len(imp_cols) > 0:
            logging.info(f' Imputation detect on {imp_cols} columns')
            act_id = self.create_activity(function_name=function_name2, used_features=list(imp_cols),
                                          description=self.description, code=self.code, code_line=self.code_line)
            self.create_relation(act_id=act_id, generated=generated_entities, used=None,
                                 invalidated=None, same=True)

        if len(trans_cols) > 0:
            logging.info(f' Value transformation detect on {trans_cols} columns')
            trans_cols = trans_cols.union(used_features)
            extra_used_entities.extend(used_entities)
            act_id = self.create_activity(function_name=function_name1, used_features=list(trans_cols),
                                          description=self.description, code=self.code, code_line=self.code_line)
            self.create_relation(act_id=act_id, generated=generated_entities,
                                 used=used_entities if len(extra_used_entities) == 0 else extra_used_entities,
                                 invalidated=None, same=len(extra_used_entities) == len(used_entities))
        else:
            logging.info(f' Value transformation and Imputation not detected')

    @suppress_tracking
    @timing
    def check_equals_dataframe(self, feature_mapping: dict):
        """
        Verifica se i dataframe df_input e df_output sono uguali.

        """

        function_name = "Check Equals Dataframe"

        result = False
        hash_df_output = self.hash_df_output.copy()
        hash_df_output.rename(feature_mapping)
        if self._df_output is not None and self.hash_df_output is not None:
            result = self.hash_df_input.equals(hash_df_output)

        if result:
            logging.info(f' {function_name}: dataframe are equals!')

        return result

    def _get_provenance_change_type(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> None:
        """
        TODO
        Cattura la provenance relativa ad un cambiamento di tipo.
        """
        int_cols = df_output.columns.intersection(df_input.columns)
        int_rows = df_output.index.intersection(df_input.index)
        values_output = df_output.to_numpy()

        generated_entities = []
        used_entities = []
        used_cols = set()

        for index in int_rows:
            i = df_output.index.get_loc(index)

            for col_name in int_cols:
                col = int_cols.get_loc(col_name)
                output_value = values_output[i][col]
                entity = self.index_col_to_input_entities.get((index, col_name), None)
                if entity:
                    if entity['value'] == output_value and entity['type'] != type(output_value):
                        used_entities.append(entity['id'])
                        generated_entities = self.create_entity()

    @suppress_tracking
    @timing
    def _get_prov_reindexing(self, df_output: pd.DataFrame) -> None:
        """
        Cattura la provenance relativa ad una reindicizzazione.

        """

        logging.info(f' Check for reindexing')

        function_name = 'Reindexing'

        index_col_to_input_entities = {}

        generated_entities = []
        used_entities = []
        used_cols = set()

        records = df_output.to_dict('index')

        for index, row in records.items():

            hash_row = sum([hash(str(e)) for e in row.values()])
            set_of_indexes = self.hash_rows_to_indexes.get(hash_row, set())

            if index not in set_of_indexes:
                for col_name in df_output.columns:
                    value = row[col_name]
                    generated_entity = None

                    for i in set_of_indexes:
                        used_entity = self.index_col_to_input_entities.get((i, col_name), None)
                        if used_entity:
                            if generated_entity is None:
                                generated_entity = self.create_entity(value=value, feature_name=col_name, index=index,
                                                                      instance=self.operation_number)
                                index_col_to_input_entities[(index, col_name)] = generated_entity
                                used_entities.append(used_entity['id'])
                                generated_entities.append(generated_entity['id'])
                                used_cols.add(col_name)
                                self.create_derivation(used_ent=used_entity['id'], gen_ent=generated_entity['id'])
                                break

        if len(used_cols) > 0:
            print(self.index_col_to_input_entities)
            self.index_col_to_input_entities.update(index_col_to_input_entities)
            print(self.index_col_to_input_entities)
            act_id = self.create_activity(function_name=function_name, used_features=list(used_cols),
                                          description=self.description, code=self.code, code_line=self.code_line)

            self.create_relation(act_id=act_id, generated=generated_entities, used=used_entities,
                                 invalidated=used_entities, same=True)

            logging.info(f' {function_name} Reindexing detected')
        else:
            logging.info(f' {function_name} Reindexing not detected')

    def get_prov_instance_generation(self, df_input, df_output):
        """
        Cattura la provenance relativa all'operazione di instance generation.

        :param df_input: df_input
        :param df_output: df_output
        :return: None

        """

        logging.info(f' Check for new instance ...')

        function_name = "Instance Generation"

        generated_entities = []
        used_cols = set()

        n_input_rows, _ = df_input.shape
        n_output_rows, _ = df_output.shape

        generated_indexes = df_output.index.difference(df_input.index)

        int_columns = df_output.columns.intersection(df_input.columns)

        for index in generated_indexes:

            output_row = df_output.iloc[index]

            for col_name in int_columns:
                output_value = output_row[col_name]
                # if not pd.isnull(new_value):
                entity = self.create_entity(value=output_value, feature_name=col_name, index=index,
                                            instance=self.operation_number)
                self.index_col_to_input_entities[(index, col_name)] = entity
                # self.index_col_to_output_entities[(index, col_name)] = entity
                generated_entities.append(entity['id'])
                used_cols.add(col_name)

        act_id = self.create_activity(function_name=function_name, used_features=list(used_cols),
                                      description=self.description, code=self.code, code_line=self.code_line)

        self.create_relation(act_id=act_id, generated=generated_entities, used=None, invalidated=None, same=False)

    @suppress_tracking
    @timing
    def generate_dataframe_metadata(self, df: pd.DataFrame, opts: dict = None):
        """

        """

        # options elaboration

        if opts is None:
            opts = {'n_bin': 10}

        n_bin = opts.get('n_bin', None)
        if n_bin is None:
            n_bin = 10

        uniques = opts.get('uniques', None)
        if uniques is None:
            uniques = True

        # generate metadata

        id = 'df_' + str(self.operation_number)
        features = list(df.columns)
        n_index = len(df.index)
        corr_matrix = df.corr().values.tolist()
        percent_missing = df.isnull().sum() * 100 // len(df)
        percent_missing = percent_missing.to_dict()
        percent_d_types = df.dtypes.value_counts() * 100 / df.shape[1]
        nuniques = df.nunique().to_dict()
        describe = df.describe().to_dict()
        df_metadata = {'id': id, 'features': features, 'n_index': n_index, 'corr_matrix': corr_matrix,
                       'percent_missing': percent_missing, 'percent_d_types': percent_d_types}

        features_metadata = {}
        for col_name in features:
            feature_metadata = {}
            feature_metadata['describe'] = describe[col_name]
            feature_metadata['nuniques'] = nuniques[col_name]
            if uniques:
                feature_metadata['unique_values'] = df[col_name].unique().tolist()
            feature_metadata['distribution'] = pd.cut(df[col_name], bins=n_bin).astype(
                str).value_counts().sort_index().to_dict()
            features_metadata[col_name] = feature_metadata

        for col_name in features:
            print(features_metadata[col_name])

    def create_activity(self, function_name, used_features=None, description=None, other_attributes=None,
                        generated_features=None, generated_records=None, deleted_used_features=None,
                        deleted_records=None, code=None,
                        code_line=None):
        """Create a provenance activity and add to the current activities array.
        Return the id of the new prov activity."""
        # Get default activity attributes:
        attributes = {}
        attributes['function_name'] = function_name
        if used_features is not None:
            attributes['used_features'] = used_features
        if description is not None:
            attributes['description'] = description
        if generated_features is not None:
            attributes['generated_features'] = generated_features
        if generated_records is not None:
            attributes['generated_records'] = generated_records
        if deleted_used_features is not None:
            attributes['deleted_used_features'] = deleted_used_features
        if deleted_records is not None:
            attributes['deleted_records'] = deleted_records
        if code is not None:
            attributes['code'] = code
        if code_line is not None:
            attributes['code_line'] = code_line
        attributes['operation_number'] = str(self.operation_number)
        # Join default and extra attributes:
        if other_attributes is not None:
            attributes.update(other_attributes)

        act_id = self.NAMESPACE_FUNC + str(uuid.uuid4())

        # Add activity to current provenance document:
        attributes['id'] = act_id

        # act = {'identifier': act_id, 'attributes': attributes}
        self.current_activities.append(attributes)

        return act_id

    def create_entity(self, value, feature_name, index, instance):
        """
        Create a provenance entity.
        Return a dictionary with the id and the record_id of the entity.
        """

        entity_id = str(value) + self.SEPARATOR + type(value).__name__ + self.SEPARATOR + str(
            feature_name) + self.SEPARATOR + str(index) + self.SEPARATOR + str(
            instance)

        instances = [instance]

        entity = {
            'id': entity_id,
            'value': value,
            'type': type(value).__name__,
            'feature_name': feature_name,
            'index': index,
            'instance': instances
        }

        self.current_entities.append(entity)
        return entity

    def create_relation(self, act_id, generated=None, used=None, invalidated=None, same=False):
        """
        Add a relation to the current relations array.
                Return the new relation.
                """
        if generated is None:
            generated = []
        if used is None:
            used = []
        if invalidated is None:
            invalidated = []
        if same:
            invalidated = []
        self.current_relations.append((generated, used, invalidated, same, act_id))

    def create_derivation(self, used_ent, gen_ent, add=True):
        """
        Add a derivation to the current relations array.
        """
        derivation = {'gen': gen_ent, 'used': used_ent}
        if add:
            self.current_derivations.append(derivation)
        return derivation

    @suppress_tracking
    @timing
    def init_prov_entities(self) -> None:
        """
        Iniziliazza le entities del dataframe in input.

        - self.index_col_to_input_entities: Crea un dizionario che ha come chiave l'indice della riga e la feature
                per consentire l'accesso diretto alle entità appena create.
        - self.hash_rows_to_indexes: Crea un dizionario inverso che ha come chiave l'hash della riga
                e che permetta di accedere alla lista di indici corrispondenti.

        return: None
        """

        self.index_col_to_input_entities = {}

        records = self.df_input.to_dict('index')
        for index, row in records.items():
            for col in self.df_input.columns:
                self.index_col_to_input_entities[(index, col)] = self.create_entity(value=row[col],
                                                                                    feature_name=col, index=index,
                                                                                    instance=self.operation_number)

            hash_row = sum([hash(str(e)) for e in row.values()])
            set_of_indexes = self.hash_rows_to_indexes.get(hash_row, set())
            set_of_indexes.add(index)
            self.hash_rows_to_indexes[hash_row] = set_of_indexes

        self.hash_df_input = self.df_input.copy()
        self.hash_df_input = self.hash_df_input.apply(
            lambda x: sum([hash(str(convert_to_int_no_decimal(e))) for e in x.values]), axis=0)

    def init_second_df(self, df_input: pd.DataFrame, index_col_to_input_entities, hash_rows_to_indexes):
        records = df_input.to_dict('index')

        for index, row in records.items():
            for col in df_input.columns:
                index_col_to_input_entities[(index, col)] = self.create_entity(value=row[col],
                                                                               feature_name=col, index=index,
                                                                               instance=self.operation_number)

            hash_row = sum([hash(str(e)) for e in row.values()])
            set_of_indexes = hash_rows_to_indexes.get(hash_row, set())
            set_of_indexes.add(index)
            hash_rows_to_indexes[hash_row] = set_of_indexes

    def prepare_for_next_operation(self, update_df_input: bool = True, save: bool = True) -> None:
        """
        Prepara per l'operazione successiva.

        """
        self.description = None

        self.df_input = self.create_tracked_dataframe(self._df_output)

        if update_df_input:
            self.df_input_copy = self.df_input.copy()
            self.hash_df_input = self.hash_df_output

        self.hash_df_output = None

        self.m_input, self.n_shape = self.df_input.shape
        self.m_output, self.n_output = self._df_output.shape

        if save:

            # TODO
            # Salvare su disco

            # Save to neo4j

            session = self.neo4j.create_session()
            self.neo4j.create_constraint(session=session)
            self.neo4j.add_activities(self.current_activities, session)
            self.neo4j.add_entities(self.current_entities)
            self.neo4j.add_derivations(self.current_derivations)
            self.neo4j.add_relations(self.current_relations)

            if self.last_activities:
                next_operations = [{'act_in_id': a_in['id'], 'act_out_id': a_out['id']} for a_out in
                                   self.current_activities
                                   for a_in in self.last_activities]
                self.neo4j.add_next_operations(next_operations, session)

            self.index_col_to_input_entities.update(self.index_col_to_output_entities.copy())
            self.index_col_to_output_entities = {}

            self.last_activities = self.current_activities.copy()

            # Free memory
            del self.current_activities[:]
            del self.current_entities[:]
            del self.current_derivations[:]
            del self.current_relations[:]

            """if self.reindexing:
                logging.info(f' detect reindexing operation: start init_prov_entities')
                self.init_prov_entities()
                self.reindexing = False"""

    def create_tracked_dataframe(self, df: pd.DataFrame):
        """
       Crea un dataframe tracciato.

        """

        class DataFrameTracked(pd.DataFrame, metaclass=ChildMeta, tracker=self):
            pass

        return DataFrameTracked(df)


class ChildMeta(type):
    """
    Definisce la metaclasse per il DataFrameTraked
    
    """

    def __new__(cls, name, bases, dct, tracker):
        """
       Ogni metodo (tranne le eccezioni) sarà incapsulato.
       La funzione wrapper si occuperà di tracciare la provenance
        
        """

        child = super().__new__(cls, name, bases, dct)

        exceptions = ['__init__', '_constructor_sliced', '_get_item_cache', '_clear_item_cache', '_ixs',
                      '_box_col_values', 'iterrows', '__repr__', '_info_repr', 'to_string', '__len__', 'itertuples',
                      'to_dict', '__getitem__', '_maybe_cache_changed', '_append', '_set_item', '_sanitize_column',
                      '_ensure_valid_index', '_set_item_mgr', '_iset_item_mgr', '_cmp_method', '_dispatch_frame_op',
                      '_construct_result', '_setitem_frame', 'isna', 'to_numpy', 'values', 'corr', 'isnull', 'nunique',
                      'select_dtypes', 'items']

        others = ['closure', 'parallel_apply',
                  'apply_parallel']

        # exceptions = ['__init__', '_constructor_sliced', '_get_item_cache', '_clear_item_cache']

        exceptions.extend(others)

        for base in bases:
            for field_name, field in base.__dict__.items():
                if callable(field):
                    if field_name not in exceptions and not isinstance(field, Iterable):
                        setattr(child, field_name, tracker._wrapper_track_provenance(field))

        return child
