from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Define the endpoint
sparql = SPARQLWrapper("https://dbpedia.org/sparql")


# We ask for the events, their date, and a quick recap
query = """
SELECT ?event ?label ?date ?abstract
WHERE {
    ?event rdf:type dbo:Event .
    ?event dbo:date ?date .
    ?event rdfs:label ?label .
    ?event dbo:abstract ?abstract .
    FILTER (?date >= "1900-01-01"^^xsd:date && ?date <= "2000-12-31"^^xsd:date) .
    FILTER (lang(?label) = "en" && lang(?abstract) = "en")
}
LIMIT 10000
"""

# Define the query
sparql.setQuery(query)

# Set return format
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Print results
# for result in results["results"]["bindings"]:
#     print(f"Event: {result['label']['value']} - Date: {result['date']['value']}, Abstract: {result['abstract']['value']}")

events_with_abstracts = [
    {
        "event": result["event"]["value"],
        "label": result["label"]["value"],
        "date": result["date"]["value"],
        "abstract": result["abstract"]["value"],
    }
    for result in results["results"]["bindings"]
]

# Save to CSV for offline use
df = pd.DataFrame(events_with_abstracts)
df.to_csv("historical_events_with_abstracts.csv", index=False)
