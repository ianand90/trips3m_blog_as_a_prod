import sys
import csv
import json
import os

from tagging import clean_html, get_header_words, compile_keywords, get_snippet_subject, \
	get_tag_values_for_snippet, get_snippet_location


def process_snippet_data(snippets):
	for snippet in snippets:
		snippet['h1_tag'] = clean_html(snippet['h1_tag'])
		snippet['h2_tag'] = clean_html(snippet['h2_tag'])
		snippet['h3_tag'] = clean_html(snippet['h3_tag'])
		snippet['h1_words'] = get_header_words(snippet['h1_tag'])
		snippet['h2_words'] = get_header_words(snippet['h2_tag'])
		snippet['h3_words'] = get_header_words(snippet['h3_tag'])


def get_grouped_tags(tags):
	grouped_tags = {}
	for tag in tags:
		tag_type = tag['tag_type']
		if tag_type in grouped_tags:
			grouped_tags[tag_type].append(tag['tag_value'].strip().lower())
		else:
			grouped_tags[tag_type] = [tag['tag_value'].strip().lower()]
	return grouped_tags


def get_tag_id_from_key_value(tags, key, value):
	for tag in tags:
		if tag['tag_type'] == key and tag['tag_value'].strip().lower() == value:
			return tag['id']


def get_grouped_subject_keywords(subjects):
	grouped_subjects = {}
	for subject in subjects:
		snippet_type = subject['snippet_type']
		if snippet_type in grouped_subjects:
			grouped_subjects[snippet_type].append(subject['keyword'].strip().lower())
		else:
			grouped_subjects[snippet_type] = [subject['keyword'].strip().lower()]
	return grouped_subjects


def read_words_to_exclude(file_path):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	with open(os.path.join(dir_path, file_path), newline='') as f:
		csv_reader = csv.DictReader(f)
		return [row['words'] for row in csv_reader]


def read_csv(file_path):
	with open(file_path, newline='') as f:
		csv_reader = csv.DictReader(f)
		return [row for row in csv_reader]


if __name__ == '__main__':
	# print("Hello there! Tighten your seat belt.")
	snippets = json.loads(sys.argv[1])
	# snippets = read_csv('data/snippets.csv')
	subject_keywords = json.loads(sys.argv[2])
	# subject_keywords = read_csv('data/subjects.csv')
	tag_type_values = json.loads(sys.argv[3])
	# tag_type_values = read_csv('data/tag_values.csv')
	post_tags = json.loads(sys.argv[4])
	# post_tags = {x['post_id']: x['tags'] for x in read_csv('data/tags.csv')}
	exclude_words = read_words_to_exclude('word_freq.csv')
	# exclude_words = [x['words'] for x in read_csv('data/word_freq.csv')]
	# print("About to take off!")

	process_snippet_data(snippets)
	subjects = get_grouped_subject_keywords(subject_keywords)
	key_vals = get_grouped_tags(tag_type_values)
	post_tags = {key: [x.strip() for x in val.split(',')] for key, val in post_tags.items() if val}

	things_to_do = compile_keywords(subjects['things_to_do'])
	places_to_visit = compile_keywords(subjects['places_to_visit'])
	accommodation = compile_keywords(subjects['accommodation'])
	things_to_do_p1 = compile_keywords(subjects['things_to_do_p1'])
	places_to_visit_p1 = compile_keywords(subjects['places_to_visit_p1'])

	key_vals = {key: compile_keywords(val) for key, val in key_vals.items()}

	results = []

	# print("We are about to airborne.")

	for snippet in snippets:
		data = {}
		data['snippet_id'] = snippet['snippet_id']
		data['snippet_type'] = get_snippet_subject(things_to_do, places_to_visit, accommodation, things_to_do_p1, places_to_visit_p1, snippet)
		data['tagged_details'] = []

		for key, val in key_vals.items():
			if key in ['Place Type', 'Activity Type']:
				values = get_tag_values_for_snippet(val, snippet, False)
			else:
				values = get_tag_values_for_snippet(val, snippet, True)

			for x in values:
				id = get_tag_id_from_key_value(tag_type_values, key, x)
				if id:
					data['tagged_details'].append(id)

		snippet_tags = post_tags.get(snippet['post_id'], [])
		location = get_snippet_location(snippet, exclude_words, snippet_tags, key_vals['Activity Type'],
										key_vals['Place Type'], things_to_do, places_to_visit, accommodation,
										things_to_do_p1, places_to_visit_p1)
		data['sub_dest'] = location[0]
		data['main_dest'] = location[1]
		results.append(data)

	# print("Take off completed!")
	print(json.dumps(results))

	# with open('data/result.csv', 'w', newline='') as f:
	# 	x = csv.DictWriter(f, fieldnames=['snippet_id', 'snippet_type', 'tagged_details', 'sub_dest', 'main_dest'])
	# 	x.writeheader()
	# 	x.writerows(results)
