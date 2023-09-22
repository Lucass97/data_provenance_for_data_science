import argparse

from prov_acquisition.repository.neo4j import Neo4jFactory
from misc.logger import CustomLogger
from misc.print_records import print_records, print_records_triplets
from misc.colors import Colors


class SimpleClient:
    """
    A simple client for interacting with a Neo4j database.

    :param str neo4j_uri: The URI for the Neo4j database.
    :param str neo4j_user: The Neo4j username for authentication.
    :param str neo4j_pwd: The Neo4j password for authentication.

    :ivar neo4j._GraphDatabaseDriver _neo4j: The Neo4j database driver instance.
    :ivar CustomLogger _logger: The custom logger for logging.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str) -> None:
        """
        Initializes the class instance with Neo4j database connection details.

        :param neo4j_uri: The URI for the Neo4j database.
        :param neo4j_user: The Neo4j username for authentication.
        :param neo4j_pwd: The Neo4j password for authentication.
        """
        self._neo4j = Neo4jFactory.create_neo4j_queries(uri=neo4j_uri,
                                                        user=neo4j_user,
                                                        pwd=neo4j_pwd)
        self._logger = CustomLogger(self.__class__)

    def all_transformations(self, tracker_id: str) -> None:
        """
        PQ1 - Retrieve all transformations for a given tracker ID.

        :param tracker_id: The ID of the tracker to retrieve transformations for.
        """

        self._logger.info(f'{Colors.BLUE}PQ1 - All-transformation query{Colors.RESET}')
        self._logger.info(f'The {tracker_id} dataframe feature was modified by the following entities:')

        records = self._neo4j.all_transformations(tracker_id)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def why_provenance(self, entity_id: str) -> None:
        """
        PQ2 - Retrieve why-provenance information for a given entity ID.

        :param entity_id: The ID of the entity to retrieve why-provenance for.
        """

        self._logger.info(f'{Colors.BLUE}PQ2 - Why-provenance query{Colors.RESET}')
        self._logger.info(f'The {entity_id} entity was influenced by the following entities:')

        records = self._neo4j.why_provenance(entity_id)
        print_records(logger=self._logger, records=records, key='m', title='Entity')

    def how_provenance(self, entity_id: str) -> None:
        """
        PQ3 - Retrieve how-provenance information for a given entity ID.

        :param entity_id: The ID of the entity to retrieve how-provenance for.
        """

        self._logger.info(f'{Colors.BLUE}PQ3 - How-provenance query{Colors.RESET}')
        self._logger.info(f'The {entity_id} entity was created by the following activities:')

        records = self._neo4j.how_provenance(entity_id=entity_id)

        print_records(logger=self._logger, records=records, key='m', title='Entity')
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def dataset_level_feature_operation(self, feature: str) -> None:
        """
        PQ4 - Retrieve dataset-level feature operation information for a given feature.

        :param feature: The name of the feature to retrieve operations for.
        """

        self._logger.info(f'{Colors.BLUE}PQ4 - Dataset-level Feature Operation query{Colors.RESET}')
        self._logger.info(f'The {feature} feature was used by the following activities:')

        records = self._neo4j.dataset_level_feature_operation(feature=feature)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def record_operation(self, index: int) -> None:
        """
        PQ5 - Retrieve record operation information for a given record index.

        :param index: The index of the record to retrieve operations for.
        """

        self._logger.info(f'{Colors.BLUE}PQ5 - Record Operation query{Colors.RESET}')
        self._logger.info(f'The {index} record was used by the following activities:')

        records = self._neo4j.record_operation(index=index)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def item_level_feature_operation(self, entity_id: str) -> None:
        """
        PQ6 - Retrieve item-level feature operation information for a given entity ID.

        :param entity_id: The ID of the entity to retrieve operations for.
        """

        self._logger.info(f'{Colors.BLUE}PQ6 - Item-level Feature Operation query{Colors.RESET}')
        self._logger.info(f'The {entity_id} entity was used by the following activities:')

        records = self._neo4j.item_level_feature_operation(entity_id=entity_id)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def feature_invalidation(self, feature: str) -> None:
        """
        PQ7 - Retrieve feature invalidation information for a given feature.

        :param feature: The name of the feature to retrieve invalidation information for.
        """

        self._logger.info(f'{Colors.BLUE}PQ7 - Feature invalidation query{Colors.RESET}')
        self._logger.info(f'The {feature} feature was invalidated by the following activities:')

        records = self._neo4j.feature_invalidation(feature=feature)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def record_invalidation(self, index: int) -> None:
        """
        PQ8 - Retrieve record invalidation information for a given record index.

        :param index: The index of the record to retrieve invalidation information for.
        """

        self._logger.info(f'{Colors.BLUE}PQ8 - Record invalidation query{Colors.RESET}')
        self._logger.info(f'The record identified by index {index} was invalidated by the following activities:')

        records = self._neo4j.record_invalidation(index=index)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def item_invalidation(self, entity_id: str) -> None:
        """
        PQ9 - Retrieve item invalidation information for a given entity ID.

        :param entity_id: The ID of the entity to retrieve invalidation information for.
        """

        self._logger.info(f'{Colors.BLUE}PQ9 - Item invalidation query{Colors.RESET}')
        self._logger.info(f'The entity {entity_id} was invalidated by the following activities:')

        records = self._neo4j.item_invalidation(entity_id=entity_id)
        print_records(logger=self._logger, records=records, key='a', title='Activity')

    def item_history(self, entity_id: str) -> None:
        """
        PQ10 - Retrieve item history information for a given entity ID.

        :param entity_id: The ID of the entity to retrieve history information for.
        """

        self._logger.info(f'{Colors.BLUE}PQ10 - Item History query{Colors.RESET}')
        self._logger.info(f'The history related to the entity {Colors.YELLOW}{entity_id}{Colors.RESET} is as follows:')

        records = self._neo4j.item_history(entity_id)

        print_records_triplets(logger=self._logger, records=records, key="r", highlights={'id': entity_id},
                               graph_file="item_history.html")

    def record_history(self, index: int) -> None:
        """
        PQ11 - Retrieve record history information for a given record index.

        :param index: The index of the record to retrieve history information for.
        """

        self._logger.info(f'{Colors.BLUE}PQ11 - Record History query{Colors.RESET}')
        self._logger.info(f'The history related to the record {Colors.YELLOW}{index}{Colors.RESET} is as follows:')

        records = self._neo4j.record_history(index=index)
        print_records_triplets(logger=self._logger, records=records, key="r", highlights={'id': index},
                               graph_file="record_history.html")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neo4j Query Tool")

    # Optional Neo4j configuration parameters
    parser.add_argument("--uri", default="bolt://localhost", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--pwd", default="adminadmin", help="Neo4j password")

    subparsers = parser.add_subparsers(dest="command", help="Specify the query command")

    # Subparser for the "all-transformations" command
    all_transformations_parser = subparsers.add_parser("all-transformations",
                                                       help="Set of operations applied to dataset and the features "
                                                            "they affect")
    all_transformations_parser.add_argument("--tracker-id", required=True, help="Tracker ID for the specified dataset")

    # Subparser for the "why-provenance" command
    why_provenance_parser = subparsers.add_parser("why-provenance",
                                                  help="Return the input data that influenced the specified entity.")
    why_provenance_parser.add_argument("--entity-id", required=True, help="Entity ID for the why-provenance query")

    # Subparser for the "how-provenance" command
    how_provenance_parser = subparsers.add_parser("how-provenance")
    how_provenance_parser.add_argument("--entity-id", required=True, help="Entity ID for the how-provenance query")

    # Subparser for the "dataset-level-feature-operation" command
    dataset_level_feature_operation_parser = subparsers.add_parser("dataset-level-feature-operation")
    dataset_level_feature_operation_parser.add_argument("--feature", required=True,
                                                        help="Feature name for the dataset-level-feature-operation "
                                                             "query")

    # Subparser for the "record-operation" command
    record_operation_parser = subparsers.add_parser("record-operation")
    record_operation_parser.add_argument("--index", required=True, type=int,
                                         help="Index for the record-operation query")

    # Subparser for the "record-invalidation" command
    record_invalidation_parser = subparsers.add_parser("record-invalidation")
    record_invalidation_parser.add_argument("--index", required=True, type=int,
                                            help="Index for the record-invalidation query")

    # Subparser for the "item-invalidation" command
    item_invalidation_parser = subparsers.add_parser("item-invalidation")
    item_invalidation_parser.add_argument("--entity-id", required=True,
                                          help="Entity ID for the item-invalidation query")

    # Subparser for the "item-operation" command
    item_level_feature_operation = subparsers.add_parser("item-level-feature-operation")
    item_level_feature_operation.add_argument("--entity-id", required=True,
                                              help="Entity ID for the item-level-feature-operation query")

    # Subparser for the "item-history" command
    item_history = subparsers.add_parser("item-history")
    item_history.add_argument("--entity-id", required=True, help="Entity ID for the item-operation query")

    # Subparser for the "record-history" command
    record_history = subparsers.add_parser("record-history")
    record_history.add_argument("--index", required=True, type=int, help="Index for the record-history query")

    # Subparser for the "feature-invalidation" command
    feature_invalidation_parser = subparsers.add_parser("feature-invalidation")
    feature_invalidation_parser.add_argument("--feature", required=True,
                                             help="Feature name for the feature-invalidation query")

    # Subparser for the "record-operation" command
    record_operation_parser = subparsers.add_parser("record-operation")
    record_operation_parser.add_argument("--index", required=True, type=int,
                                         help="Index for the record-operation query")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Create a Neo4j connection with the provided configuration parameters
    client = SimpleClient(neo4j_uri=args.uri,
                          neo4j_user=args.user, neo4j_pwd=args.pwd)

    # Call the corresponding function based on the chosen subcommand
    if args.command == "all-transformations":
        client.all_trasformations(args.tracker_id)
    elif args.command == "why-provenance":
        client.why_provenance(args.entity_id)
    elif args.command == "how-provenance":
        client.how_provenance(args.entity_id)
    elif args.command == "dataset-level-feature-operation":
        client.dataset_level_feature_operation(args.feature)
    elif args.command == "record-operation":
        client.record_operation(args.index)
    elif args.command == "record-invalidation":
        client.record_invalidation(args.index)
    elif args.command == "item-invalidation":
        client.item_invalidation(args.entity_id)
    elif args.command == "item-level-feature-operation":
        client.item_level_feature_operation(args.entity_id)
    elif args.command == "item-history":
        client.item_history(args.entity_id)
    elif args.command == "record-history":
        client.record_history(args.index)
    elif args.command == "feature-invalidation":
        client.feature_invalidation(args.feature)


if __name__ == '__main__':
    main()
