from rmp_client import RMPClient

# schools to look up
school_names = [
    'University of Rochester',
    'Rochester Institute of Technology',
    'SUNY Geneseo',
    'SUNY Brockport',
    'Nazareth University',
    'St. John Fisher University',
]


# turns whatever object the package returns into a plain dictionary
def to_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, 'model_dump'):
        return item.model_dump()
    return dict(vars(item))


with RMPClient() as client:
    for name in school_names:
        print("\n=====", name, "=====")
        results = client.search_schools(name)

        # some versions return a list, others return an object holding the list
        if not isinstance(results, list):
            results = to_dict(results).get('schools', results)

        # print the top 3 matches so the correct school (and state) can be checked by hand
        for school in list(results)[:3]:
            print(to_dict(school))