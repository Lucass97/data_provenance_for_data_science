import argparse
from numpy import str0

from tabulate import tabulate
from pyvis.network import Network

from prov_acquisition.repository.neo4j import Neo4jFactory
from misc.logger import CustomLogger
from misc.print_records import print_records
from misc.colors import Colors


class SimpleClient:

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str) -> None:
        
        self.__neo4j = Neo4jFactory.create_neo4j_queries(uri=neo4j_uri,
                                                         user=neo4j_user,
                                                         pwd=neo4j_pwd)
        self.__logger = CustomLogger(self.__class__)
    
    def all_trasformations(self, tracker_id: str) -> None:
        """
        PQ1
        """

        self.__logger.info(f'Selected {tracker_id} dataframe')

        records = self.__neo4j.all_transformations(tracker_id)
        
        for i, record in enumerate(records):
            table_info = [[f'Activity #{i + 1}', '']]
            activity_data = record.data()['a']
            for key in activity_data:
                table_info.append([key, activity_data[key]])
            
            activity_info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
            
            self.__logger.info(f'\n{activity_info}')

    def why_provenance(self, entity_id: str) -> None:
        """
        PQ2
        """

        self.__logger.info(f'Selected {entity_id}')

        records = self.__neo4j.why_provenance(entity_id)
        
        for i, record in enumerate(records):
            table_info = [[f'Entity #{i + 1}', '']]
            entity_data = record.data()['m']
            for key in entity_data:
                table_info.append([key, entity_data[key]])
            
            activity_info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
            
            self.__logger.info(f'\n{activity_info}')
    
    def how_provenance(self, entity_id: str) -> None:
        """
        PQ3
        """

        self.__logger.info(f'Selected {entity_id}')

        records = self.__neo4j.how_provenance(entity_id=entity_id)
        
        print_records(logger=self.__logger, records=records, key='m', title='Entity')
        print_records(logger=self.__logger, records=records, key='a', title='Activity')
    

    def dataset_level_feature_operation(self, feature:str) -> None:
        """
        PQ4
        """

        self.__logger.info(f'Selected feature {feature}')

        records = self.__neo4j.dataset_level_feature_operation(feature=feature)

        print_records(logger=self.__logger, records=records, key='a', title='Activity')

    
    def record_operation(self, index: int) -> None:
        """
        PQ5
        """

        self.__logger.info(f'Selected {index} index')

        records = self.__neo4j.record_operation(index=index)
        
        print_records(logger=self.__logger, records=records, key='a', title='Activity')

    def item_level_feature_operation(self, entity_id: str) -> None:
        """
        PQ6
        """
        self.__logger.info(f'Selected {entity_id} enttity')

        records = self.__neo4j.item_level_feature_operation(entity_id=entity_id)
        
        print_records(logger=self.__logger, records=records, key='a', title='Activity')

   
    def feature_invalidation(self, feature: str) -> None:
        """
        PQ7
        """
        self.__logger.info(f'PQ7 - Feature invalidation query')
        self.__logger.info(f'La {feature} feature è stata invalidata dalle seguenti attività:')

        records = self.__neo4j.feature_invalidation(feature=feature)
        
        print_records(logger=self.__logger, records=records, key='a', title='Activity')
     
    def record_invalidation(self, index: int) -> None:
        """
        PQ8
        """
        self.__logger.info(f'PQ8 - Record invalidation query')
        self.__logger.info(f'Il record identificato dall indice {index} è stato invalidato dalle seguenti attività:')

        records = self.__neo4j.record_invalidation(index=index)
        
        print_records(logger=self.__logger, records=records, key='a', title='Activity')
    
    def item_invalidation(self, entity_id: str) -> None:
        """
        PQ9
        """
        self.__logger.info(f'PQ9 - Item invalidation query')
        self.__logger.info(f'L entita {entity_id} è stata invalidata dalle seguenti attività:')

        records = self.__neo4j.item_invalidation(entity_id=entity_id)
        
        print_records(logger=self.__logger, records=records, key='a', title='Activity')
    
    
    def item_history(self, entity_id: str) -> None:

        net = Network(notebook=True)

        self.__logger.info(f'{Colors.BLUE}PQ10 - Item History query{Colors.RESET}')
        self.__logger.info(f'L history relativa all entita {Colors.YELLOW}{entity_id}{Colors.RESET} è la seguente:')

        records = self.__neo4j.item_history(entity_id)
        
        table_info = [[f'{Colors.GREEN}Source Entity', 'Relationship', f'Target Entity{Colors.RESET}']]
        for i, record in enumerate(records):
            
            activity = record.data()
            for triple in activity["r"]:
                source_node = triple[0]
                relationship = triple[1]
                target_node = triple[2]

                if source_node['id'] == entity_id:
                    source_node['id'] = f"{Colors.YELLOW}{source_node['id']}{Colors.RESET}"
                
                if target_node['id'] == entity_id:
                    target_node['id'] = f"{Colors.YELLOW}{source_node['id']}{Colors.RESET}"

                
                table_info.append([source_node['id'], relationship, target_node['id']])
                
                net.add_node(source_node['id'], label=source_node['id'])
                net.add_node(target_node['id'], label=target_node['id'])
                net.add_edge(source_node['id'], target_node['id'], label=relationship)
            
        info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
        self.__logger.info(f'\n{info}')
        
        self.__logger.info('è stato generato un file html contente il grafo.')
        net.show("item_history.html")

    def record_history(self, index: int) -> None:

        net = Network(notebook=True)

        self.__logger.info(f'{Colors.BLUE}PQ11 - Record History Query{Colors.RESET}')
        self.__logger.info(f'L history relativa al record {Colors.YELLOW}{index}{Colors.RESET} è la seguente:')

        
        records = self.__neo4j.record_history(index=index)
        
        #print_records(logger=self.__logger, records=records, key='m', title='Entity')
        table_info = [[f'{Colors.GREEN}Source Entity', 'Relationship', f'Target Entity{Colors.RESET}']]
        for record in records:
            
            activity = record.data()
            for triple in activity["r"]:
                
                source_node = triple[0]
                relationship = triple[1]
                target_node = triple[2]

                table_info.append([source_node['id'], relationship, target_node['id']])
                
                net.add_node(source_node['id'], label=source_node['id'])
                net.add_node(target_node['id'], label=target_node['id'])
                net.add_edge(source_node['id'], target_node['id'], label=relationship)
            
        info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
        self.__logger.info(f'\n{info}')
        
        self.__logger.info('è stato generato un file html contente il grafo.')
        net.show("item_history.html")
            
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neo4j Query Tool")
    
    # Optional Neo4j configuration parameters
    parser.add_argument("--uri", default="bolt://localhost", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--pwd", default="adminadmin", help="Neo4j password")
    
    subparsers = parser.add_subparsers(dest="command", help="Specify the query command")

    # Subparser for the "all-transformations" command
    all_transformations_parser = subparsers.add_parser("all-transformations", help="Set of operations applied to dataset and the features they affect")
    all_transformations_parser.add_argument("--tracker-id", required=True, help="Tracker ID for the specified dataset")
    
    # Subparser for the "why-provenance" command
    why_provenance_parser = subparsers.add_parser("why-provenance", help="Return the input data that influenced the specified entity.")
    why_provenance_parser.add_argument("--entity-id", required=True, help="Entity ID for the why-provenance query")
    
    # Subparser for the "how-provenance" command
    how_provenance_parser = subparsers.add_parser("how-provenance")
    how_provenance_parser.add_argument("--entity-id", required=True, help="Entity ID for the how-provenance query")

    # Subparser for the "dataset-level-feature-operation" command
    dataset_level_feature_operation_parser = subparsers.add_parser("dataset-level-feature-operation")
    dataset_level_feature_operation_parser.add_argument("--feature", required=True, help="Feature name for the dataset-level-feature-operation query")
    
    # Subparser for the "record-operation" command
    record_operation_parser = subparsers.add_parser("record-operation")
    record_operation_parser.add_argument("--index", required=True, type=int, help="Index for the record-operation query")

    # Subparser for the "record-invalidation" command
    record_invalidation_parser = subparsers.add_parser("record-invalidation")
    record_invalidation_parser.add_argument("--index", required=True, type=int, help="Index for the record-invalidation query")
    
    # Subparser for the "item-invalidation" command
    item_invalidation_parser = subparsers.add_parser("item-invalidation")
    item_invalidation_parser.add_argument("--entity-id", required=True, help="Entity ID for the item-invalidation query")

    # Subparser for the "item-operation" command
    item_level_feature_operation = subparsers.add_parser("item-level-feature-operation")
    item_level_feature_operation.add_argument("--entity-id", required=True, help="Entity ID for the item-level-feature-operationquery")

    # Subparser for the "item-history" command
    item_history = subparsers.add_parser("item-history")
    item_history.add_argument("--entity-id", required=True, help="Entity ID for the item-operation query")

    # Subparser for the "record-history" command
    record_history = subparsers.add_parser("record-history")
    record_history.add_argument("--index", required=True, type=int, help="Index for the record-history query")
    

    # Subparser for the "feature-invalidation" command
    feature_invalidation_parser = subparsers.add_parser("feature-invalidation")
    feature_invalidation_parser.add_argument("--feature", required=True, help="Feature name for the feature-invalidation query")

    # Subparser for the "record-operation" command
    record_operation_parser = subparsers.add_parser("record-operation")
    record_operation_parser.add_argument("--index", required=True, type=int, help="Index for the record-operation query")

    
    return parser

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    
    # Create a Neo4j connection with the provided configuration parameters
    client = SimpleClient(neo4j_uri=args.uri, neo4j_user=args.user, neo4j_pwd=args.pwd)
    
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