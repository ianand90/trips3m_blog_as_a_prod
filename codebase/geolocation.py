import csv
import requests

from copy import copy

from utils import get_config


def read_data_from_csv(file_path):
	rows = []
	print("Reading CSV file.")
	with open(file_path, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			rows.append({
				'location': row['main_dest'],
				'snippets': row['snippets']
			})
	print("Done reading CSV file.")
	return rows


def create_permutations(items, left, right, result):
	"""
	finds out all the permutations of data present in items and append it to result.
	:param items: list containing data to permute
	:param left: left index of items
	:param right: right index if items
	:param result: list which will contain all the permutations
	:return: None
	"""
	if left == right:
		result.append(copy(items))
	else:
		for i in range(left, right + 1):
			items[left], items[i] = items[i], items[left]
			create_permutations(items, left + 1, right, result)
			items[left], items[i] = items[i], items[left]  # backtrack


def get_request_param(search_query):
	"""
	returns the search param to be used for requesting google places API
	:param search_query: string representing place to search
	:return: dict object containing query param
	"""
	return {
		# 'input': search_query,
		# 'inputtype': 'textquery',
		# 'fields': get_config('google_api', 'fields'),
		'address': search_query,
		'key': get_config('google_api', 'api_key')
	}


def get_location_search_results(search_query):
	"""
	sends HTTP request to google places API to get the search results
	:param search_query: string representing place to search
	:return: search results
	"""
	try:
		api_url = get_config('google_api', 'api_url')
		request_params = get_request_param(search_query)
		print("Making HTTP request to Google places API...")
		response = requests.get(api_url, params=request_params)
		print("Response received from Google place API...")
		return response.json()
	except Exception as e:
		raise Exception("Error encountered while requesting Google API: {0}".format(e))


def get_search_queries(snippet_data):
	"""
	creates a list of search queries from snippet data
	:param snippet_data: {
		'location': ['location'],
		'snippets': [[id, ['sub', 'location']], [id, ['sub', 'location']], ...]
	}
	:return: list of search result
	"""
	try:
		location = ' '.join(snippet_data['location'])
		sub_locations = []

		for snippet in snippet_data['snippets']:
			create_permutations(snippet[1], 0, len(snippet[1])-1, sub_locations)

		for index, sub_location in enumerate(sub_locations):
			sub_locations[index] = ', '.join([' '.join(sub_location), location])

		return sub_locations
	except Exception as e:
		raise Exception("Provided input is not a valid snippet data. {0}.".format(e))


def fetch_search_result_for_queries(queries):
	"""
	fetch search results for all the search queries
	:param queries: list of search terms
	:return: None
	"""
	assert type(queries) is list
	result = []
	for query in queries:
		response = get_location_search_results(query)
		result.append(response)
	return result


def get_address_part(add_type, components):
	for component in components:
		if add_type in component['types']:
			return component['long_name']


def test():
	def test_permutations():
		result = []
		items = ['A', 'B', 'C']
		create_permutations(items, 0, len(items) - 1, result)
		print(result)

	def test_query_param():
		print(get_request_param('hello'))

	def test_search_results():
		import json
		import ast
		import random

		search_queries = []

		with open('data/result/result_mapped_dest.csv', newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				sub_dests = ast.literal_eval(row['sub_dest'])
				for sub_dest in sub_dests:
					if sub_dest and sub_dest[1]:
						search_queries.append([', '.join([' '.join(sub_dest[1]), row['main_dest']]), row['main_dest'], sub_dest, sub_dest[0]])

		# picked_index = []
		# picked_queries = []
		# total = len(search_queries)
		# while len(picked_queries) < 1000:
		# 	index = random.randint(0, total-1)
		# 	if index not in picked_index:
		# 		picked_index.append(index)
		# 		picked_queries.append(search_queries[index])

		# print(len(picked_queries))
		picked_queries = search_queries
		print(len(picked_queries))

		with open('data/result/result_location.csv', 'w', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=['snippet_id', 'main_dest', 'sub_dest', 'search_term', 'formatted_address', 'lat', 'lng', 'country', 'pincode', 'state', 'district'])
			writer.writeheader()
			for index, query in enumerate(picked_queries):
				result = get_location_search_results(query[0])
				data = {
					'main_dest': query[1],
					'sub_dest': query[2],
					'search_term': query[0],
					'snippet_id': query[3]
				}
				try:
					data['formatted_address'] = result['results'][0]['formatted_address']
					data['lat'] = result['results'][0]['geometry']['location']['lat']
					data['lng'] = result['results'][0]['geometry']['location']['lng']
					data['country'] = get_address_part('country', result['results'][0]['address_components'])
					data['state'] = get_address_part('administrative_area_level_1', result['results'][0]['address_components'])
					data['district'] = get_address_part('administrative_area_level_2', result['results'][0]['address_components'])
					data['pincode'] = get_address_part('postal_code', result['results'][0]['address_components'])
				except Exception as e:
					print(str(e))
				print("writing row: {0}.".format(index))
				writer.writerow(data)

	test_search_results()


if __name__ == '__main__':
	test()
