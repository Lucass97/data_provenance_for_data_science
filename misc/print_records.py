from tabulate import tabulate

def print_records(logger, records, key, title) -> str:

    for i, record in enumerate(records):
            table_info = [[f'{title} #{i + 1}', '']]
            print(record.data())
            element = record.data().get(key, None)
            print(element)
            if element is None:
                 continue

            for key in element:
                table_info.append([key, element[key]])
            
            info = tabulate(table_info, headers="firstrow", tablefmt="fancy_grid")
            
            logger.info(f'\n{info}')
