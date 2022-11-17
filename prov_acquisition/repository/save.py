import json


class Saver:
    ACTIVITIES_NAMESPACE = 'ACTIVITIES'
    ENTITIES_NAMESPACE = 'ENTITIES'
    DERIVATIONS_NAMESPACE = 'DERIVATIONS'
    RELATIONS_NAMESPACE = 'RELATIONS'

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.current_instance = 0

    def save_on_disk(self, activities, entities, derivations, relations):
        with open(self.base_path + '/' + self.current_instance + '/' + self.ACTIVITIES_NAMESPACE + '.json', 'w') as f:
            json.dump(activities, f)

        with open(self.base_path + '/' + self.current_instance + '/' + self.ENTITIES_NAMESPACE + '.json', 'w') as f:
            json.dump(entities, f)

        with open(self.base_path + '/' + self.current_instance + '/' + self.DERIVATIONS_NAMESPACE + '.json', 'w') as f:
            json.dump(derivations, f)

        with open(self.base_path + '/' + self.current_instance + '/' + self.RELATIONS_NAMESPACE + '.json', 'w') as f:
            json.dump(relations, f)

        self.current_instance += 1
