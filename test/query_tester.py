import argparse
import inspect
from typing import Callable

import sys

sys.path.append("../")

from misc.print_records import print_records
from prov_acquisition import constants
from client.simple_client import SimpleClient


class QueryTester(SimpleClient):
    """
    A class for testing Neo4j queries.

    :param str neo4j_uri: The Neo4j server URI.
    :param str neo4j_user: The Neo4j username.
    :param str neo4j_pwd: The Neo4j password.
    :param int limit: The limit for random entity selection. Default is 3.

    :ivar list __random_entities: List of randomly selected entities.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str, limit: int = 3) -> None:
        """
        Initializes a QueryTester instance.

        :param str neo4j_uri: The Neo4j server URI.
        :param str neo4j_user: The Neo4j username.
        :param str neo4j_pwd: The Neo4j password.
        :param int limit: The limit for random entity selection. Default is 3.
        """
        super().__init__(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_pwd=neo4j_pwd)

        # Get random entities from Neo4j
        self.__random_entities = self._neo4j.get_random_nodes(label=constants.ENTITY_LABEL, limit=limit,
                                                              session=self._session)
        self._logger.info('Random selected entities: ')
        print_records(records=self.__random_entities, logger=self._logger, key='n', title="Entity")

        # Iterate through inherited methods and override each method to take entity_id
        for method_name in dir(self):
            method = getattr(self, method_name)
            if callable(method) and not method_name.startswith("__") and method_name in dir(SimpleClient):
                setattr(self, method_name, self.__execute_method_multiple_times(method))

    def __execute_method_multiple_times(self, original_method: Callable) -> Callable:
        """
        Modifies a method to execute multiple times with different entity_id.

        :param Callable original_method: The original method to modify.

        :return: The modified method.
        :rtype: Callable
        """
        params = inspect.signature(original_method).parameters

        param_mapping = {
            'entity_id': 'id',
            'index': 'index',
            'feature': 'feature_name'
        }

        def modified_method(*args, **kwargs):
            results = []
            for record in self.__random_entities:
                element = record['n']
                mapped_kwargs = {param: element[param_mapping[param]]
                                 for param in params if param in param_mapping}
                result = original_method(*args, **mapped_kwargs, **kwargs)
                results.append(result)

            return results

        return modified_method


def create_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the Neo4j Query Tester.

    :return: The argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Neo4j Query Tester")

    # Optional configuration parameters for Neo4j
    parser.add_argument("--uri", default="bolt://localhost", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--pwd", default="adminadmin", help="Neo4j password")
    parser.add_argument("--limit", default=3, help="Random entities to obtain")

    # Positional argument for the command
    parser.add_argument("command", choices=["all-transformations", "why-provenance", "how-provenance",
                                            "dataset-level-feature-operation", "record-operation",
                                            "record-invalidation", "item-invalidation",
                                            "item-level-feature-operation", "item-history",
                                            "record-history", "feature-invalidation",
                                            "feature-spread", "dataset-spread"], help="Specify the command")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Create a Neo4j connection with the provided configuration parameters
    client = QueryTester(neo4j_uri=args.uri, neo4j_user=args.user, neo4j_pwd=args.pwd, limit=args.limit)

    # Call the corresponding function based on the chosen subcommand
    if args.command == "all-transformations":
        client.all_transformations()
    elif args.command == "why-provenance":
        client.why_provenance()
    elif args.command == "how-provenance":
        client.how_provenance()
    elif args.command == "dataset-level-feature-operation":
        client.dataset_level_feature_operation()
    elif args.command == "record-operation":
        client.record_operation()
    elif args.command == "record-invalidation":
        client.record_invalidation()
    elif args.command == "item-invalidation":
        client.item_invalidation()
    elif args.command == "item-level-feature-operation":
        client.item_level_feature_operation()
    elif args.command == "item-history":
        client.item_history()
    elif args.command == "record-history":
        client.record_history()
    elif args.command == "feature-invalidation":
        client.feature_invalidation()
    elif args.command == "feature-spread":
        client.feature_spread()
    elif args.command == "dataset-spread":
        client.dataset_spread()


if __name__ == '__main__':
    main()
