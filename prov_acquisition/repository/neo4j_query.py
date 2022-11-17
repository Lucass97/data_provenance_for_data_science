import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import Optional

import neo4j
from neo4j import GraphDatabase

from misc.utils import timing


class Neo4jConnection:
    """
    Classe che definisce un connettore per Neo4j
    """

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query: str, parameters: dict = None, db: str = None, session: neo4j.Session = None) -> Optional[
        list]:

        """
        Esegue una query. Se la sessione passata risulta nulla allora ne verrà creata una internamente alla funzione per poter eseguire la query.

        :param query: Query da eseguire
        :param parameters: Parametri per la query
        :param db: database da eseguire.
        :param session: sessione neo4j creata esternamente da usare per eseguire la query.
        """

        assert self.__driver is not None, "Driver not initialized!"
        external_session = False
        if session:
            external_session = True
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None and not external_session:
                session.close()
        return response

    def create_session(self, db=None):
        """
        Crea una sessione per neo4j

        :return: una sessione neo4j.
        """
        return self.__driver.session(database=db) if db is not None else self.__driver.session()

    @timing
    def create_constraint(self, session=None):

        query = """DROP CONSTRAINT ON (a:Activity) ASSERT a.id IS UNIQUE"""
        self.query(query=query, parameters=None, session=session)

        query = """DROP CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE"""
        self.query(query=query, parameters=None, session=session)

        query = """CREATE CONSTRAINT constraint_activity_id ON (a:Activity) ASSERT a.id IS UNIQUE"""
        self.query(query=query, parameters=None, session=session)

        query = """CREATE CONSTRAINT constraint_entity_id ON (e:Entity) ASSERT e.id IS UNIQUE"""
        self.query(query=query, parameters=None, session=session)

    @timing
    def delete_all(self, session=None):

        query = '''
            MATCH(n)
            DETACH
            DELETE
            n;
        '''
        return self.query(query, parameters=None, session=session)

    @timing
    def add_activities(self, activities, session=None):

        query = '''
            UNWIND $rows AS row
            CREATE (a:Activity)
            SET a = row
            
        '''
        return self.query(query, parameters={'rows': activities}, session=session)

    @timing
    def add_entities(self, entities):

        query = '''
                UNWIND $rows AS row
                CREATE (e:Entity)
                SET e=row
                '''
        return self.insert_data_multiprocess(query=query, rows=entities)

    def udpate_entities(self, entities):
        query = '''
                UNWIND $rows AS row
                MATCH (e:Entity)
                WHERE e.id = row.id
                SET e=row
                '''
        return self.insert_data_multiprocess(query=query, rows=entities)

    @timing
    def add_derivations(self, derivations):

        query = '''
                   UNWIND $rows AS row
                   MATCH (e1:Entity {id: row.gen})
                   WITH e1, row
                   MATCH (e2:Entity {id: row.used})
                   MERGE (e1)-[:WAS_DERIVED_FROM]->(e2)
                    '''
        return self.insert_data_multiprocess(query=query, rows=derivations)

    @timing
    def add_relations(self, relations):

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
                    MATCH (e:Entity {id: row})
                    WITH e
                    MATCH (a:Activity {id: $act_id})
                    MERGE (a)-[:USED]->(e)
                    '''
            query2 = '''
                    UNWIND $rows AS row
                    MATCH (e:Entity {id: row})
                    WITH e
                    MATCH (a:Activity {id: $act_id})
                    MERGE (e)-[:WAS_GENERATED_BY]->(a)
                    '''
            query3 = '''
                    UNWIND $rows AS row
                    MATCH (e:Entity {id: row})
                    WITH e
                    MATCH (a:Activity {id: $act_id})
                    MERGE (e)-[:WAS_INVALIDATED_BY]->(a)
                    '''

            self.insert_data_multiprocess(query=query1, rows=used, act_id=act_id)
            self.insert_data_multiprocess(query=query2, rows=generated, act_id=act_id)
            self.insert_data_multiprocess(query=query3, rows=invalidated, act_id=act_id)

    @timing
    def add_next_operations(self, next_operations, session=None):
        query = ''' 
                                   UNWIND $next_operations AS next_operation
                                   MATCH (a1:Activity {id: next_operation.act_in_id})
                                   WITH a1, next_operation
                                   MATCH (a2:Activity {id: next_operation.act_out_id})
                                   MERGE (a1)-[:NEXT]->(a2)
                                   
                                '''

        self.query(query, parameters={'next_operations': next_operations}, session=session)

    def insert_data(self, query, rows, batch_size=100, session=None):
        # Function to handle the updating the Neo4j database in batch mode.

        pool = Pool(processes=(cpu_count() - 1))

        total = 0
        batch = 0
        start = time.time()
        result = None

        while batch * batch_size < len(rows):
            """res = self.query(query,
                             parameters={'rows': rows[batch * batch_size:(batch + 1) * batch_size]}, session=session)
            rameters = {'rows': rows[batch * batch_size:(batch + 1) * batch_size]}
            batch += 1
            result = {"total": total,
                      "batches": batch,
                      "time": time.time() - start}
            print(result)"""
            p = {'rows': rows[batch * batch_size:(batch + 1) * batch_size]}
            print(len(p))
            # pool.apply_async(self.query, (query,), dict(parameters=p))

        pool.close()
        pool.join()

    def insert_data_multiprocess(self, query: str, rows, **kwargs) -> None:
        """
        Divide in batch i dati. Ciascun batch è assegnato ad un processo che si occuperà di caricarlo su neo4j.
        Il metodo termina quando tutti i worker hanno concluso la loro esecuzione.

        :param query: query da eseguire
        :param rows: righe da caricare.
        :kwargs: eventuali parametri aggiuntivi da caricare.

        :return: None
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
