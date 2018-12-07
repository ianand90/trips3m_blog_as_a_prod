import csv
import requests
from collections import defaultdict

from utils import get_config


def get_request_header():
	return {
		'Content-Type': 'application/x-www-form-urlencoded',
		'Authorization': 'Basic {0}'.format(get_config('snippet_api', 'auth_key'))
	}


def post_data_to_api(file_path, headers, tag_key, csv_header='types'):
	with open(file_path, newline='') as f:
		csv_reader = csv.DictReader(f)
		for row in csv_reader:
			data = {
				'tag_key': tag_key,
				'tag_value': ' '.join([x.capitalize() for x in row[csv_header].split('-')])
			}
			r = requests.post(get_config('snippet_api', 'key_val_post_api'), headers=headers, data=data)
			print(r.text)


def add_new_key_val():
	headers = get_request_header()
	post_data_to_api('data/input/places_types.csv', headers, 'Place Type')
	post_data_to_api('data/input/activity_type.csv', headers, 'Activity Type', 'activity_type')
	post_data_to_api('data/input/month_types.csv', headers, 'Month')
	post_data_to_api('data/input/season_types.csv', headers, 'Season')
	post_data_to_api('data/input/with_whom_types.csv', headers, 'With Whom')
	post_data_to_api('data/input/occassion_types.csv', headers, 'Occasion')
	post_data_to_api('data/input/timeofday_type.csv', headers, 'Time Of The Day')
	post_data_to_api('data/input/budget_type.csv', headers, 'Budget')
	post_data_to_api('data/input/theme_type.csv', headers, 'Theme')

	for val in ['Things to do', 'Places to Visit', 'Accommodation']:
		data = {
			'tag_key': 'Snippet Type',
			'tag_value': val
		}
		r = requests.post(get_config('snippet_api', 'key_val_post_api'), headers=headers, data=data)
		print(r.text)


def fetch_key_vals():
	r = requests.get(get_config('snippet_api', 'key_val_get_api'), headers={
		'Authorization': 'Basic {0}'.format(get_config('snippet_api', 'auth_key'))
	})
	import re
	import json
	res = re.sub(r'<!.*?>$', '', r.text)
	response = json.loads(res)

	data = {}
	for x in response:
		if x['tag_type'] in data:
			data[x['tag_type']].append({
				'id': x['id'],
				'tag_value': x['tag_value']
			})
		else:
			data[x['tag_type']] = [{
				'id': x['id'],
				'tag_value': x['tag_value']
			}]
	return data


key_vals = fetch_key_vals()


def get_key_val_ids_from_values(key, values):
	assert type(values) is list
	ids = []
	vals = key_vals[key]
	for val in values:
		if val:
			val = val if val == 'any' else val[4:]
			found = False
			for v in vals:
				if v['tag_value'] == ' '.join([x.capitalize() for x in val.split('-')]):
					ids.append(v['id'])
					found = True
			if not found:
				print("Can not find id for Key: '{0}' and Value: '{1}'.".format(key, val))
				# raise Exception("Can not find id for Key: '{0}' and Value: '{1}'.".format(key, val))

	return ids


def get_tagged_details(row):
	return ','.join(
		get_key_val_ids_from_values('Place Type', [x.strip() for x in row['place_tags'].split(',')]) +
		get_key_val_ids_from_values('Activity Type', [x.strip() for x in row['activity_tags'].split(',')]) +
		get_key_val_ids_from_values('Month', [x.strip() for x in row['month_tags'].split(',')]) +
		get_key_val_ids_from_values('Season', [x.strip() for x in row['season_tags'].split(',')]) +
		get_key_val_ids_from_values('With Whom', [x.strip() for x in row['with_whom_tags'].split(',')]) +
		get_key_val_ids_from_values('Occasion', [x.strip() for x in row['occassion_tags'].split(',')]) +
		get_key_val_ids_from_values('Time Of The Day', [x.strip() for x in row['time_of_day_tags'].split(',')]) +
		get_key_val_ids_from_values('Budget', [x.strip() for x in row['budget_tags'].split(',')]) +
		get_key_val_ids_from_values('Theme', [x.strip() for x in row['theme_tags'].split(',')])
	)


def get_tag_values_for_snippets():
	snippets = {}
	with open('data/result/result_new.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			snippets[row['snippet_id']] = {
				'snippetType': row['subject_tag'],
				'taggedDetails': get_tagged_details(row)
			}
	return snippets


def get_location_for_snippets():
	snippets = {}
	with open('data/result/result_location.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			snippets[row['snippet_id']] = {
				'location': row['search_term'],
				'formatedAddress': row['formatted_address'],
				'locationLat': row['lat'],
				'locationLong': row['lng'],
				'country': row['country'],
				'state': row['state'],
				'district': row['district'],
				'pincode': row['pincode']
		}
	return snippets


def create_post_data():
	tag_data = get_tag_values_for_snippets()
	location_data = get_location_for_snippets()
	result =  defaultdict(dict)
	for data in (tag_data, location_data):
		for key, val in data.items():
			result[key].update(val)
	return result


def post_snippet_data():
	headers = get_request_header()
	data = create_post_data()
	for key, val in data.items():
		val['snippetId'] = key
		r = requests.post(get_config('snippet_api', 'snippet_update_api'), headers=headers, data=val)
		print(r.text)


def test():
	# add_new_key_val()
	post_snippet_data()


if __name__ == '__main__':
	test()