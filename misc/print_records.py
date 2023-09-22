from tabulate import tabulate
from pyvis.network import Network

from misc.colors import Colors


def print_records(logger, records, key, title) -> None:
    if len(records) == 0:
        logger.info(f'No data')
        return

    for i, record in enumerate(records):

        table_info = [[f'{title} #{i + 1}', '']]

        element = record.data().get(key, None)

        if element is None:
            continue

        for prop in element:
            table_info.append([prop, element[prop]])

        info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")

        logger.info(f'\n{info}')


def print_records_triplets(logger, records, key: str, highlights: dict, graph_file: str) -> None:
    net = Network(notebook=True)

    if len(records) == 0:
        logger.info(f'No data')
        return

    table_info = [[f'{Colors.GREEN}Source Entity',
                   'Relationship', f'Target Entity{Colors.RESET}']]
    for record in records:

        activity = record.data()
        for triple in activity[key]:
            source_node = triple[0]
            relationship = triple[1]
            target_node = triple[2]

            for prop in highlights:
                if source_node[prop] == highlights[prop]:
                    source_node['id'] = f"{Colors.YELLOW}{source_node['id']}{Colors.RESET}"

                if target_node[prop] == highlights[prop]:
                    target_node['id'] = f"{Colors.YELLOW}{source_node['id']}{Colors.RESET}"

            table_info.append(
                [source_node['id'], relationship, target_node['id']])

            net.add_node(source_node['id'], label=source_node['id'])
            net.add_node(target_node['id'], label=target_node['id'])
            net.add_edge(source_node['id'],
                         target_node['id'], label=relationship)

    info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
    logger.info(f'\n{info}')

    logger.info('è stato generato un file html contente il grafo.')

    net.show(graph_file)
