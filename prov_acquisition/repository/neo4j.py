from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import List, Optional

import prov_acquisition.constants as constants
from misc.decorators import timing, Singleton
from misc.logger import CustomLogger
from neo4j import GraphDatabase, Session


@Singleton
class Neo4jConnector:
    """
    Class defining a connector for Neo4j.
    """

    def __init__(self, uri: str, user: str, pwd: str) -> None:
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        self.__logger = CustomLogger('ProvenanceTracker')

        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            self.__logger.error('Failed to create the driver:', e)

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """

        if self.__driver is not None:
            self.__driver.close()

    def create_session(self, db=None) -> Session:
        """
        Creates a Neo4j session.

        :param db: Optional parameter specifying the database to connect to.
        :return: A Neo4j session.
        """

        return self.__driver.session(database=db) if db is not None else self.__driver.session()


class Neo4jQueryExecutor:
    """
    Class that executes queries for Neo4j.
    """

    def __init__(self, connector) -> None:
        self.__connector = connector
        self.__logger = CustomLogger('ProvenanceTracker')

    def query(self, query: str, parameters: dict = None, db: str = None, session: Session = None) -> Optional[list]:
        """
        Executes a query. If the provided session is None, it creates a new session internally to execute the query.

        :param query: The query to execute.
        :param parameters: Parameters for the query.
        :param db: The database to connect to.
        :param session: An externally created Neo4j session to use for executing the query.
        :return: The query result as a list or None if an error occurred.
        """

        if not self.__connector:
            raise ValueError('Connector not initialized!')

        response = None

        external_session = False
        if session:
            external_session = True

        try:
            if not external_session:
                session = self.__connector.create_session(db=db)
            response = session.run(query, parameters).data()
        except Exception as e:
            self.__logger.error(f'Query failed: {e} {query}')
        finally:
            # Close the session if it was internally created
            if session is not None and not external_session:
                session.close()
        return response

    def insert_data_multiprocess(self, query: str, rows: List[any], **kwargs) -> None:
        """
        Divides the data into batches. Each batch is assigned to a process that loads it into Neo4j.
        The method completes when all workers have finished execution.

        :param query: The query to execute.
        :param rows: The rows to load.
        :kwargs: Additional parameters to load.
        """

        pool = Pool(processes=(cpu_count() - 1))
        batch_size = len(rows) // (cpu_count() - 1) if len(rows) >= cpu_count() - 1 else cpu_count()
        batch = 0

        while batch * batch_size < len(rows):
            parameters = {'rows': rows[batch * batch_size:(batch + 1) * batch_size]}
            pool.apply_async(self.query, args=(query,), kwds={'parameters': {**parameters, **kwargs}})
            batch += 1

        pool.close()
        pool.join()


class Neo4jQueries:
    """
    Class containing predefined queries for Neo4j.
    """

    def __init__(self, query_executor):
        self.__query_executor = query_executor
        self.logger = CustomLogger("ProvenanceTracker")

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def create_constraint(self, session=None) -> None:
        """
        Creates constraints for Neo4j nodes.

        :param session: An optional Neo4j session to use for executing the query.
        """

        query = '''DROP CONSTRAINT ''' + constants.ACTIVITY_CONSTRAINT + ''''''
        self.__query_executor.query(query=query, parameters=None, session=session)

        query = '''DROP CONSTRAINT ''' + constants.ENTITY_CONSTRAINT
        self.__query_executor.query(query=query, parameters=None, session=session)

        query = '''CREATE CONSTRAINT ''' + constants.ACTIVITY_CONSTRAINT + \
                ''' FOR (a:''' + constants.ACTIVITY_LABEL + \
                ''') REQUIRE a.id IS UNIQUE'''
        self.__query_executor.query(query=query, parameters=None, session=session)

        query = '''CREATE CONSTRAINT ''' + constants.ENTITY_CONSTRAINT + \
                ''' FOR (e:''' + constants.ENTITY_LABEL + \
                ''') REQUIRE e.id IS UNIQUE'''
        self.__query_executor.query(query=query, parameters=None, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def delete_all(self, session=None):
        """
        Deletes all nodes and relationships in the database.

        :param session: An optional Neo4j session to use for executing the query.
        :return: The query result as a list or None if an error occurred.
        """

        query = '''
                MATCH(n)
                DETACH
                DELETE
                n;
                '''
        self.logger.debug(msg=query)
        self.__query_executor.query(query, parameters=None, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def add_activities(self, activities: List[any], session=None) -> None:
        """
        Adds activities to the database.

        :param activities: The activities to add.
        :param session: An optional Neo4j session to use for executing the query.
        :return: The query result as a list or None if an error occurred.
        """
        query = '''
                UNWIND $rows AS row
                CREATE (a:''' + constants.ACTIVITY_LABEL + ''')
                SET a = row    
                '''
        self.logger.debug(msg=query)
        self.__query_executor.query(query, parameters={'rows': activities}, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def add_entities(self, entities: List[any]) -> None:
        """
        Adds entities to the database.

        :param entities: The entities to add.
        :return: None
        """
        query = '''
                UNWIND $rows AS row
                CREATE (e:''' + constants.ENTITY_LABEL + ''')
                SET e=row
                '''
        self.logger.debug(msg=query)
        self.__query_executor.insert_data_multiprocess(query=query, rows=entities)

    def udpate_entities(self, entities: List[any]) -> None:
        """
        Updates entities in the database.

        :param entities: The entities to update.
        :return: None
        """
        query = '''
                UNWIND $rows AS row
                MATCH (e:''' + constants.ENTITY_LABEL + ''')
                WHERE e.id = row.id
                SET e=row
                '''
        self.logger.debug(msg=query)
        self.__query_executor.insert_data_multiprocess(query=query, rows=entities)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def add_derivations(self, derivations: List[any]) -> None:
        """
        Adds derivations (relationships between entities) to the database.

        :param derivations: The derivations to add.
        :return: None
        """
        query = '''
                UNWIND $rows AS row
                MATCH (e1:''' + constants.ENTITY_LABEL + ''' {id: row.gen})
                WITH e1, row
                MATCH (e2:''' + constants.ENTITY_LABEL + ''' {id: row.used})
                MERGE (e1)-[:''' + constants.DERIVATION_RELATION + ''']->(e2)
                '''
        self.logger.debug(msg=query)
        self.__query_executor.insert_data_multiprocess(query=query, rows=derivations)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def add_relations(self, relations: List[any]) -> None:
        """
        Adds relations (relationships between activities and entities) to the database.

        :param relations: The relations to add.
        :return: None
        """
        for relation in relations:
            generated = relation[0]
            used = relation[1]
            invalidated = relation[2]
            same = relation[3]
            act_id = relation[4]

            if same:
                invalidated = used

            query1 = '''
                    UNWIND $rows AS row
                    MATCH (e:''' + constants.ENTITY_LABEL + ''' {id: row})
                    WITH e
                    MATCH (a:''' + constants.ACTIVITY_LABEL + ''' {id: $act_id})
                    MERGE (a)-[:''' + constants.USED_RELATION + ''']->(e)
                    '''
            query2 = '''
                    UNWIND $rows AS row
                    MATCH (e:''' + constants.ENTITY_LABEL + ''' {id: row})
                    WITH e
                    MATCH (a:''' + constants.ACTIVITY_LABEL + ''' {id: $act_id})
                    MERGE (e)-[:''' + constants.GENERATION_RELATION + ''']->(a)
                    '''
            query3 = '''
                    UNWIND $rows AS row
                    MATCH (e:''' + constants.ENTITY_LABEL + ''' {id: row})
                    WITH e
                    MATCH (a:''' + constants.ACTIVITY_LABEL + ''' {id: $act_id})
                    MERGE (e)-[:''' + constants.INVALIDATION_RELATION + ''']->(a)
                    '''

            self.__logger.debug(msg=query1)
            self.__logger.debug(msg=query2)
            self.__logger.debug(msg=query3)

            self.__query_executor.insert_data_multiprocess(query=query1, rows=used, act_id=act_id)
            self.__query_executor.insert_data_multiprocess(query=query2, rows=generated, act_id=act_id)
            self.__query_executor.insert_data_multiprocess(query=query3, rows=invalidated, act_id=act_id)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def add_next_operations(self, next_operations: List[any], session=None) -> None:
        """
        Adds relationships between activities representing the order in which they occur.

        :param next_operations: The next operations to add.
        :param session: An optional Neo4j session to use for executing the query.
        :return: None
        """
        query = ''' 
                UNWIND $next_operations AS next_operation
                MATCH (a1:''' + constants.ACTIVITY_LABEL + ''' {id: next_operation.act_in_id})
                WITH a1, next_operation
                MATCH (a2:''' + constants.ACTIVITY_LABEL + ''' {id: next_operation.act_out_id})
                MERGE (a1)-[:''' + constants.NEXT_RELATION + ''']->(a2)
                '''

        self.logger.debug(msg=query)

        self.__query_executor.query(
            query, parameters={'next_operations': next_operations}, session=session)

    def create_useful_indexes(self, session=None) -> None:

        query = f'''
                CREATE INDEX entity_index IF NOT EXISTS
                FOR (n:{constants.ENTITY_LABEL})
                ON (n.id)
                '''
        self.__query_executor.query(query=query, parameters=None, session=session)

        query = f'''
                CREATE LOOKUP INDEX rel_type_lookup_index
                FOR()-[r]-() ON EACH type(r)
                '''
        self.__query_executor.query(query=query, parameters=None, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def all_transformations(self, session=None):
        query = f'''
                MATCH (a:{constants.ACTIVITY_LABEL})
                RETURN a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def why_provenance(self, entity_id: str, session=None):
        query = '''  
                MATCH (e:''' + constants.ENTITY_LABEL + '''{id:"''' + entity_id + '''"})-[:''' + constants.DERIVATION_RELATION + ''']->(m:''' + constants.ENTITY_LABEL + ''')
                RETURN e,m
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def how_provenance(self, entity_id: str, session=None):
        query = '''  
                MATCH (e:''' + constants.ENTITY_LABEL + '''{id:"''' + entity_id + '''"})-[:''' + constants.DERIVATION_RELATION + ''']->(m:''' + constants.ENTITY_LABEL + ''') 
                MATCH (m:''' + constants.ENTITY_LABEL + ''')-[]-(a:''' + constants.ACTIVITY_LABEL + ''')
                return e,m,a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def dataset_level_feature_operation(self, feature: str, session=None):
        query = f'''
                MATCH (a:{constants.ACTIVITY_LABEL})
                WHERE "{feature}" IN a.used_features
                RETURN a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def record_operation(self, index: str, session=None):
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})<-[:{constants.USED_RELATION}]-(a:{constants.ACTIVITY_LABEL})
                WHERE {index} = e.index
                RETURN DISTINCT a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def item_level_feature_operation(self, entity_id: str, session=None):
        query = '''  
                MATCH (e:''' + constants.ENTITY_LABEL + '''{id:"''' + entity_id + '''"})-[]-(a:''' + constants.ACTIVITY_LABEL + ''')
                RETURN e,a
                '''
        self.logger.debug(msg=query)
        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def item_invalidation(self, entity_id: str, session=None):
        query = '''  
                MATCH (e:''' + constants.ENTITY_LABEL + '''{id:"''' + entity_id + '''"})-[:''' + constants.INVALIDATION_RELATION + ''']->(a:''' + constants.ACTIVITY_LABEL + ''')
                RETURN e,a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def feature_invalidation(self, feature: str, session=None):
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})-[:{constants.INVALIDATION_RELATION}]->(a:{constants.ACTIVITY_LABEL})
                WHERE "{feature}" IN a.used_features
                RETURN DISTINCT a
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def record_invalidation(self, index: str, session=None):
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})-[:{constants.INVALIDATION_RELATION}]->(a:{constants.ACTIVITY_LABEL})
                WHERE {index} = e.index AND a.deleted_records = true
                RETURN DISTINCT a
        '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def record_history(self, index: str, session=None):
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})-[r:{constants.DERIVATION_RELATION}*1..]-(m:{constants.ENTITY_LABEL})
                WHERE {index} = e.index
                RETURN DISTINCT e,r,m
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def item_history(self, entity_id: str, session=None):
        query = '''  
                MATCH p=(e:''' + constants.ENTITY_LABEL + '''{id:"''' + entity_id + '''"})-[r:''' + constants.DERIVATION_RELATION + '''*1..]-(m:''' + constants.ENTITY_LABEL + ''')
                RETURN DISTINCT e,r,m
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def get_random_nodes(self, label: str, limit: int = 3, session=None):
        query = f'''
                MATCH (n:{label})
                WITH n, rand() AS random
                ORDER BY random
                LIMIT {limit}
                RETURN n
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def dataset_spread(self, session=None):
        """
        returns number of invalidated and new entities for each activity -> ?
        """
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})-[r]->(a:{constants.ACTIVITY_LABEL})
                RETURN a, type(r) AS t, count(*) AS c
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)

    @timing(log_file=constants.NEO4j_QUERY_EXECUTION_TIMES)
    def feature_spread(self, feature: str, session=None):
        """
        returns number of invalidated and new entities for each activity that operates on feature -> return activity with max inv and one with max new?
        """
        query = f'''
                MATCH (e:{constants.ENTITY_LABEL})-[r]->(a:{constants.ACTIVITY_LABEL})
                WHERE "{feature}" IN a.used_features
                RETURN a, TYPE(r) AS t, COUNT(*) AS c
                '''

        self.logger.debug(msg=query)

        return self.__query_executor.query(query, session=session)


class Neo4jFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_neo4j_queries(uri: str, user: str, pwd: str) -> Neo4jQueries:
        """
        Creates Neo4jQueries object for executing queries on Neo4j.

        :param uri: The URI of the Neo4j database.
        :param user: The username for accessing the Neo4j database.
        :param pwd: The password for accessing the Neo4j database.
        :return: A Neo4jQueries object.
        """

        connector = Neo4jConnector(uri, user, pwd)
        query_executor = Neo4jQueryExecutor(connector)
        queries = Neo4jQueries(query_executor)
        return queries
