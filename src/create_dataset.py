from argparse import ArgumentParser
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from utils import print_dataset_statistics

ENTITY_TYPES = [
    "Q5",  # human
    "Q6256",  # country
    "Q515",  # city
    "Q1549591",  # big city
    "Q105543609",  # musical work/composition
    "Q215380",  # musical group
    "Q18127",  # record label
    "Q7725634",  # literary work
    "Q11424",  # film
    "Q8142",  # currency
    "Q891723",  # public company
    "Q19967801",  # online service
    "Q7889",  # video game
    "Q210167",  # video game developer
    "Q3658341",  # literary character
    "Q1667921",  # novel series
    "Q1320047",  # book publisher
    "Q1762059",  # film production company
    "Q15773347",  # film character
]

RELATION_TYPES = {
    "P27": "country of citizenship",
    "P26": "spouse",
    "P22": "father",
    "P25": "mother",
    "P36": "capital",
    "P38": "currency",
    "P1304": "central bank",
    "P85": "anthem",
    "P86": "composer",
    "P175": "performer",
    "P740": "location of formation",
    "P800": "notable work",
    "P264": "record label",
    "P57": "director",
    "P162": "producer",
    "P58": "screenwriter",
    "P272": "production company",
    "P127": "owned by",
    "P495": "country of origin",
    "P178": "developer",
    "P169": "chief executive officer",
    "P159": "headquarters location",
    "P1441": "present in work",
    "P50": "author",
    "P123": "publisher",
    "P108": "employer",
    "P35": "head of state",
    "P112": "founded by",
    "P1789": "chief operating officer",
}

RELATION_BEFORE_ENTITY_TEMPLATES = {
    "P27": "the country of citizenship of {}",
    "P26": "the spouse of {}",
    "P22": "the father of {}",
    "P25": "the mother of {}",
    "P36": "the capital of {}",
    "P38": "the currency of {}",
    "P1304": "the central bank of {}",
    "P85": "the national anthem of {}",
    "P86": "the composer of {}",
    "P175": "the performer of {}",
    "P50": "the author of {}",
    "P57": "the director of {}",
    "P162": "the producer of {}",
    "P58": "the screenwriter of {}",
    "P127": "the owner of {}",
    "P495": "the country of origin of {}",
    "P178": "the developer of {}",
    "P123": "the publisher of {}",
    "P169": "the chief executive officer of {}",
    "P159": "the headquarters location of {}",
    "P740": "the location of formation of {}",
    "P800": "the most notable work of {}",
    "P264": "the record label of {}",
    "P272": "the company that produced {}",
    "P1441": "the work that features {}",
    "P108": "the employer of {}",
    "P35": "the head of state of {}",
    "P112": "the founder of {}",
    "P1789": "the chief operating officer of {}",
}

RELATION_AFTER_ENTITY_TEMPLATES = {
    "P27": "{} holds citizenship in",
    "P26": "{} has a spouse named",
    "P22": "{} has a father named",
    "P25": "{} has a mother named",
    "P36": "{} has a capital named",
    "P38": "{} has a currency named",
    "P1304": "{} has a central bank named",
    "P85": "{} has a national anthem named",
    "P86": "{} is composed by",
    "P175": "{} is performed by",
    "P50": "{} is authored by",
    "P57": "{} is directed by",
    "P162": "{} is produced by",
    "P58": "{} has a screenwriter named",
    "P127": "{} is owned by",
    "P495": "{} has a country of origin named",
    "P178": "{} is developed by",
    "P123": "{} is published by",
    "P169": "{} is led by a chief executive officer named",
    "P159": "{} has its headquarters in",
    "P740": "{} was formed in",
    "P800": "{} is the most notable work of",
    "P264": "{} is signed to the record label",
    "P272": "{} is produced by the company",
    "P1441": "{} is featured in the work",
    "P108": "{} is employed by",
    "P35": "{} is the head of state of",
    "P112": "{} was founded by",
    "P1789": "{} is led by a chief operating officer named",
}

USER_AGENT = "'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'"
QUERY_BATCH_SIZE = 100


class SPARQLClient:

    def __init__(self):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=USER_AGENT)
        self.sparql.setReturnFormat(JSON)

    def format_value(self, value):
        parsed_uri = urlparse(value)
        if all([parsed_uri.scheme, parsed_uri.netloc]):
            return parsed_uri.path.split('/')[-1]
        return value

    def execute(self, query, format_values=True):
        self.sparql.setQuery(query)
        responses = self.sparql.query().convert()
        records = []
        for response in responses['results']['bindings']:
            record = {}
            for key in response:
                if format_values:
                    record[key] = self.format_value(response[key]['value'])
                else:
                    record[key] = response[key]['value']
            records.append(record)
        return pd.DataFrame(records)


def get_popular_entities_per_type(ent_type, k=1, offset=0):
    return f'''
        SELECT ?entity ?statementCount WHERE {{
            ?entity wdt:P31 wd:{ent_type} .
            ?entity wikibase:statements ?statementCount .
        }} 
        ORDER BY DESC(?statementCount)
        LIMIT {k} OFFSET {offset}
    '''


def get_entities_per_type(ent_type, k=1, offset=0):
    return f'''
        SELECT ?entity WHERE {{
            ?entity wdt:P31 wd:{ent_type}.
        }} 
        LIMIT {k} OFFSET {offset}
    '''


def get_one_hop(ent):
    return f'''
        SELECT ?subject ?relation ?target WHERE {{
            VALUES ?node {{<{ent}>}}
            ?subject ?wdt ?target.
            ?relation a wikibase:Property;
                wikibase:propertyType wikibase:WikibaseItem; 
                wikibase:directClaim ?wdt.
        }}
    '''


def get_two_hops(entities, k=100):
    return f'''
        SELECT ?e1 ?r1 ?e2 ?r2 ?e3 WHERE {{
            VALUES ?e2 {{{" ".join([f"wd:{e1}" for e1 in entities])}}}
            ?e1 ?wdt ?e2.
            ?r1 a wikibase:Property;
                wikibase:propertyType wikibase:WikibaseItem; 
                wikibase:directClaim ?wdt.
            
            ?e2 ?subWdt ?e3.
            ?r2 a wikibase:Property;
                wikibase:propertyType wikibase:WikibaseItem; 
                wikibase:directClaim ?subWdt.
                
            FILTER(?r1 IN ({", ".join([f"wd:{r1}" for r1 in RELATION_TYPES.keys()])}))
            FILTER(?r2 IN ({", ".join([f"wd:{r2}" for r2 in RELATION_TYPES.keys()])}))
        }}
        LIMIT {k}
    '''


def get_labels(entities):
    return f'''
        SELECT ?entity ?entityLabel {{
            VALUES ?entity {{{" ".join([f"wd:{entity}" for entity in entities])}}}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    '''


def get_entities_type(entities):
    return f'''
        SELECT ?entity ?type ?typeLabel WHERE {{
            {{VALUES ?entity {{ {" ".join([f"wd:{e}" for e in entities])} }} }}
            ?entity wdt:P31 ?type.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}        
    '''


def get_entity_aliases(entities):
    return f'''
        SELECT ?entity ?alias WHERE {{
            {{VALUES ?entity {{ {" ".join([f"wd:{e}" for e in entities])} }} }}
            ?entity skos:altLabel ?alias.
            FILTER(LANG(?alias) = "en")
        }}
        '''


def merge_attribute(two_hops, entity, attributes, attribute_column, target_suffix):
    two_hops = two_hops.merge(attributes, left_on=entity, right_on="entity", how="left")
    two_hops = two_hops.drop(columns=["entity"])
    two_hops = two_hops.rename(columns={attribute_column: f"{entity}_{target_suffix}"})
    return two_hops


def add_all_labels(client, two_hops):
    to_label = pd.concat([two_hops["e1"], two_hops["r1"], two_hops["e2"], two_hops["r2"], two_hops["e3"]]).unique()
    if len(to_label) == 0:
        raise ValueError("No entities to label")
    labels = []
    for batch_start in tqdm(range(0, len(to_label), 100), desc="Retrieving entity labels"):
        labels.append(client.execute(get_labels(to_label[batch_start:batch_start + 100])))
    labels = pd.concat(labels)
    for entity in ["e1", "r1", "e2", "r2", "e3"]:
        two_hops = merge_attribute(two_hops, entity, labels, "entityLabel", "label")
    return two_hops


def drop_non_unique(two_hops, subject, relation, target):
    return two_hops[two_hops.groupby([subject, relation])[target].transform("nunique").eq(1)]


def drop_all_non_unique(two_hops):
    two_hops = two_hops.drop_duplicates(subset=["e1", "r1", "e2", "r2", "e3"])
    two_hops = two_hops[(two_hops["e1"] != two_hops["e2"]) &
                        (two_hops["e2"] != two_hops["e3"]) &
                        (two_hops["e3"] != two_hops["e1"])]
    return two_hops


def add_all_entity_types(client, two_hops):
    to_label = pd.concat([two_hops["e1"], two_hops["e2"], two_hops["e3"]]).unique()
    types = []
    for batch_start in tqdm(range(0, len(to_label), 100), desc="Retrieving entity types"):
        types.append(client.execute(get_entities_type(to_label[batch_start:batch_start + 100])))
    types = pd.concat(types)
    types = types.replace("big city", "city")
    if types.empty:
        raise ValueError("No entity types found")
    types = types[types["type"].isin(ENTITY_TYPES)]
    types = types.drop(columns=["type"])
    types = types.drop_duplicates(subset=["entity"])
    for entity in ["e1", "e2", "e3"]:
        two_hops = merge_attribute(two_hops, entity, types, "typeLabel", "type")
    return two_hops


def add_all_relation_types(two_hops):
    two_hops["r1_type"] = two_hops["r1"].map(RELATION_TYPES)
    two_hops["r2_type"] = two_hops["r2"].map(RELATION_TYPES)
    return two_hops


def drop_invalid_entities(two_hops):
    return two_hops[(two_hops["e1"].str.startswith("Q")) &
                    (two_hops["e2"].str.startswith("Q")) &
                    (two_hops["e3"].str.startswith("Q"))]


def add_all_templates(two_hops, target):
    two_hops["r1_template"] = two_hops["r1"].map(RELATION_BEFORE_ENTITY_TEMPLATES)
    if target == "rbe":
        two_hops["r2_template"] = two_hops["r2"].map(RELATION_BEFORE_ENTITY_TEMPLATES)
    else:
        two_hops["r2_template"] = two_hops["r2"].map(RELATION_AFTER_ENTITY_TEMPLATES)
    return two_hops


def add_prompts(two_hops, target):
    if target == "rbe":
        two_hops["source_prompt"] = two_hops.apply(
            lambda row: row["r2_template"].format(row["r1_template"]).format(row["e1_label"]) + " is",
            axis=1)
    else:
        two_hops["source_prompt"] = two_hops.apply(
            lambda row: row["r2_template"].format(row["r1_template"]).format(row["e1_label"]),
            axis=1)
    return two_hops


def do_mediawiki_request(url, params, expected_response="query"):
    last_continue = {}
    while True:
        params = params.copy()
        params.update(last_continue)
        response = requests.get(url, params=params).json()
        if "error" in response:
            print(response)
            break
        if expected_response in response:
            yield response[expected_response]
        if "continue" not in response:
            break
        last_continue = response['continue']


def get_entity_wikipedia_titles(entities):
    params = {
        "action": "wbgetentities",
        "format": "json",
        "prop": "sitelinks",
        "ids": "|".join(entities),
        "sitefilter": "enwiki",
        "languages": "en",
    }
    titles = {}
    for response in do_mediawiki_request("https://www.wikidata.org/w/api.php",
                                         params=params,
                                         expected_response="entities"):
        for entity, data in response.items():
            if "sitelinks" in data and "enwiki" in data["sitelinks"]:
                titles[data["sitelinks"]["enwiki"]["title"]] = entity
    return titles


def get_entities_popularity(entities, days=7):
    views = {}
    batches = [entities[i:i + 50] for i in range(0, len(entities), 50)]
    for ent_batch in tqdm(batches, desc="Calculating popularity"):
        titles = get_entity_wikipedia_titles(ent_batch)
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageviews",
            "titles": "|".join(titles.keys()),
            "pvipdays": days,
        }
        for response in do_mediawiki_request("https://en.wikipedia.org/w/api.php", params):
            for page in response["pages"].values():
                if "pageviews" in page:
                    page_views = [views for views in page["pageviews"].values() if views is not None]
                    views[titles[page["title"]]] = int(np.mean(page_views)) if len(page_views) != 0 else 0
    return views


def add_all_entity_popularity(two_hops):
    all_entities = pd.concat([two_hops["e1"], two_hops["e2"], two_hops["e3"]]).unique()
    views = get_entities_popularity(all_entities)
    two_hops["e1_popularity"] = two_hops["e1"].map(views).fillna(0)
    two_hops["e2_popularity"] = two_hops["e2"].map(views).fillna(0)
    two_hops["e3_popularity"] = two_hops["e3"].map(views).fillna(0)
    return two_hops


def add_all_entity_aliases(client, two_hops):
    all_entities = pd.concat([two_hops["e1"], two_hops["e2"], two_hops["e3"]]).unique()
    aliases = []
    for batch_start in tqdm(range(0, len(all_entities), 20), desc="Retrieving entity aliases"):
        aliases.append(client.execute(get_entity_aliases(all_entities[batch_start:batch_start + 20])))
    aliases = pd.concat(aliases)
    if aliases.empty:
        aliases = pd.DataFrame(columns=["entity", "alias"])
    aliases = aliases.groupby("entity")["alias"].apply(list).reset_index(name="aliases")
    for entity in ["e1", "e2", "e3"]:
        two_hops = merge_attribute(two_hops, entity, aliases, "aliases", "aliases")
    return two_hops


def main(args):
    client = SPARQLClient()
    datasets = []
    for ent_type in tqdm(ENTITY_TYPES, desc="Retrieving entities by type"):
        for batch_num in tqdm(range(args.entities_per_type // QUERY_BATCH_SIZE), desc="Entity type batches",
                              leave=False):
            try:
                entities = client.execute(get_popular_entities_per_type(ent_type, k=QUERY_BATCH_SIZE,
                                                                        offset=batch_num * QUERY_BATCH_SIZE))
            except Exception:
                entities = client.execute(get_entities_per_type(ent_type, k=QUERY_BATCH_SIZE,
                                                                offset=batch_num * QUERY_BATCH_SIZE))
            if entities.empty:
                continue

            for entity in tqdm(entities["entity"], desc="Retrieving two hops", leave=False):
                target_dataset = client.execute(get_two_hops([entity]))
                if target_dataset.empty:
                    continue
                target_dataset = drop_invalid_entities(target_dataset)
                target_dataset = drop_all_non_unique(target_dataset)
                datasets.append(target_dataset)

    dataset = pd.concat(datasets)
    dataset = dataset.drop_duplicates(subset=["e1", "r1", "e2", "r2", "e3"])
    dataset = add_all_entity_aliases(client, dataset)
    dataset = add_all_entity_types(client, dataset)
    dataset = add_all_relation_types(dataset)
    dataset = add_all_labels(client, dataset)
    dataset = add_all_entity_popularity(dataset)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.rename_axis("id").reset_index()
    dataset.to_csv(f"{args.datasets_dir}/two_hop.csv")
    print_dataset_statistics(dataset)

    for target in ["rbe", "rae"]:
        target_dataset = add_all_templates(dataset, target=target)
        target_dataset = add_prompts(target_dataset, target=target)
        target_dataset = target_dataset[["id", "e1", "r1", "e2", "r2", "e3",
                                         "e1_label", "e2_label", "e3_label",
                                         "e1_aliases", "e2_aliases", "e3_aliases",
                                         "e1_type", "r1_type", "e2_type", "r2_type", "e3_type",
                                         "e1_popularity", "e2_popularity", "e3_popularity",
                                         "r1_template", "r2_template",
                                         "prompt"]]
        target_dataset.to_csv(f"{args.datasets_dir}/two_hop_{target}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--entities-per-type", type=int, default=2000)
    main(parser.parse_args())
