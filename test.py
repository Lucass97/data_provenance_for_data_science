import argparse
import inspect
from typing import Callable

from misc.print_records import print_records
from prov_acquisition import constants
from simple_client import SimpleClient


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
        self.__random_entities = self._neo4j.get_random_nodes(label=constants.ENTITY_LABEL, limit=limit)
        self._logger.info('Random selected entities: ')
        print_records(records=self.__random_entities,
                      logger=self._logger, key='n', title="Entity")

        # Iterate through inherited methods and override each method to take entity_id
        for method_name in dir(self):
            method = getattr(self, method_name)
            if callable(method) and not method_name.startswith("__") and method_name in dir(SimpleClient):
                setattr(self, method_name,
                        self.__execute_method_multiple_times(method))

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
            'feature': 'feature'
        }

        def modified_method(*args, **kwargs):
            for record in self.__random_entities:
                element = record.data()['n']
                mapped_kwargs = {param: element[param_mapping[param]]
                                 for param in params if param in param_mapping}
                return original_method(*args, **mapped_kwargs, **kwargs)

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

    # Positional argument for the command
    parser.add_argument("command", choices=["all-transformations", "why-provenance", "how-provenance",
                                            "dataset-level-feature-operation", "record-operation",
                                            "record-invalidation", "item-invalidation",
                                            "item-level-feature-operation", "item-history",
                                            "record-history", "feature-invalidation"],
                        help="Specify the command")

    return parser
