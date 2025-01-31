from serpapi import GoogleSearch
import os

def search_serpapi(q):
    params = {
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "q": q,
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": "30"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def concat_snippets(organic_results):
    # filter out results without a snippet
    organic_results = [result for result in organic_results if 'snippet' in result]
    # filter out results from NBC News
    organic_results = [result for result in organic_results if 'NBC' not in result['source'] and 'NBC' not in result['title']]
    # filter out links containing 'fact'
    organic_results = [result for result in organic_results if 'fact' not in result['link']]
    organic_results = organic_results[:5]
    return '\n'.join([f'Title: {result["title"]}\nSource: {result["source"]}, {result["date"] if "date" in result else ""}\nContent: {result["snippet"]}' for result in organic_results])

def get_google_ctx(q):
    search_results = search_serpapi(q)
    if 'organic_results' in search_results:
        return concat_snippets(search_results['organic_results'])
    else:
        return ""