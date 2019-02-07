import nltk
import re

from collections import OrderedDict
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup


# nltk.download('rslp')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def lemma(word):
	porter = PorterStemmer()
	snowball = SnowballStemmer("english")
	wordnet = WordNetLemmatizer()

	ls = [porter.stem(word),snowball.stem(word),wordnet.lemmatize(word)]

	curr_set = set(ls)

	ls = list(curr_set)
	for w in ls:
		if w == word:
			ls.remove(w)
	return ls


def get_word_forms(word):
	forms = [word]
	root_words = lemma(word)
	for root_word in root_words:
		if root_word:
			forms.append(root_word)
	return forms


def is_stop_word(word):
	stop_words = set(stopwords.words('english'))
	return word in stop_words


def is_punc(word):
	punc_list = set(["-", ":", ".", "!", "_", "'", "&", "...", ","])
	return word in punc_list


def is_stop_word_or_punc(word):
	return is_stop_word(word) or is_punc(word)


def compile_keywords(keywords):
	words = {}

	for word in keywords:
		word = word.strip().lower()
		if not is_stop_word_or_punc(word):
			if ' ' in word:
				parts = word.split()
				last_part_forms = get_word_forms(parts.pop())
				words[word] = []

				for form in last_part_forms:
					new_parts = parts + [form]
					words[word].append("".join(new_parts))
					words[word].append(new_parts)

			else:
				words[word] = get_word_forms(word)

	return words


def clean_html(raw_html):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', raw_html)
	cleanr2 = re.compile('{.*?}')
	cleantext2 = re.sub(cleanr2, '', cleantext)
	return cleantext2


def get_tokenized_words(header):
	return word_tokenize(str(header).lower().strip())


def get_header_words(header):
	words = []
	for word in get_tokenized_words(header):
		words += get_word_forms(word)
	return set(words)


def is_match(tag_values, headers):
	assert type(tag_values) is list
	assert type(headers) is set

	try:  # is single word type
		val = set(tag_values)
		if headers & val:
			return True

	except TypeError:  # is multiple word type
		for val in tag_values:
			if type(val) is list:
				val = set(val)
				if len(headers & val) == len(val):
					return True
			else:
				if val in headers:
					return True
	return False


def get_snippet_subject(things_to_do, places_to_visit, accommodation, things_to_do_p1, places_to_visit_p1, snippet):
	if not snippet['h3_tag']:
		for key, val in things_to_do.items():
			if is_match(val, snippet['h1_words']):
				return 'Things To Do'
		for key, val in places_to_visit.items():
			if is_match(val, snippet['h1_words']):
				return 'Places To Visit'
		for key, val in accommodation.items():
			if is_match(val, snippet['h1_words']):
				return 'Accommodation'
		for key, val in things_to_do_p1.items():
			if is_match(val, snippet['h1_words']):
				return 'Things To Do'
		for key, val in places_to_visit_p1.items():
			if is_match(val, snippet['h1_words']):
				return 'Places To Visit'
	else:
		for key, val in things_to_do.items():
			if is_match(val, snippet['h2_words']) or is_match(val, snippet['h1_words']):
				return 'Things To Do'
		for key, val in places_to_visit.items():
			if is_match(val, snippet['h2_words']) or is_match(val, snippet['h1_words']):
				return 'Places To Visit'
		for key, val in accommodation.items():
			if is_match(val, snippet['h2_words']) or is_match(val, snippet['h1_words']):
				return 'Accommodation'
		for key, val in things_to_do_p1.items():
			if is_match(val, snippet['h2_words']) or is_match(val, snippet['h1_words']):
				return 'Things To Do'
		for key, val in places_to_visit_p1.items():
			if is_match(val, snippet['h2_words']) or is_match(val, snippet['h1_words']):
				return 'Places To Visit'


def get_tag_and_header_intersection(tags, headers):
	assert type(tags) is dict
	assert type(headers) is set

	matches = []

	for key, val in tags.items():
		if is_match(val, headers):
			matches.append(key)

	return matches


def get_tag_values_for_snippet(tags, snippet, has_h3_check=True):
	if has_h3_check:
		if not snippet['h3_tag']:
			return get_tag_and_header_intersection(tags, snippet['h1_words']) or ['any']
		else:
			return get_tag_and_header_intersection(tags, snippet['h2_words']) or \
				   get_tag_and_header_intersection(tags, snippet['h1_words']) or ['any']

	else:
		return get_tag_and_header_intersection(tags, snippet['h3_words']) or \
			   get_tag_and_header_intersection(tags, snippet['h2_words']) or \
			   get_tag_and_header_intersection(tags, snippet['h1_words']) or ['any']


def get_image_alt(image_tag):
	soup = BeautifulSoup(image_tag)
	if soup.img:
		return soup.img['alt']


def clean_alt_text(alt_text):
	alt_text = re.sub(r'[\n0-9_]', '', alt_text)
	return alt_text.replace("-", " ")


def get_words_from_text(text):
	words = get_tokenized_words(text)
	return [x for x in words if not is_stop_word_or_punc(x)]


def get_ordered_words(order_like, words):
	assert type(order_like) is list

	if len(words) < 2:
		return words

	occurrence = {order_like.index(word): word for word in words}
	od = OrderedDict(sorted(occurrence.items()))
	return list(od.values())


def get_words_to_exclude(activity_type, place_type, things_to_do, places_to_visit, accommodation,
						 things_to_do_p1, places_to_visit_p1, exclude_words):
	def extract_words(collection):
		words = []
		for val in collection.values():
			try:
				set(val)
				words += val
			except TypeError:
				for x in val:
					if type(x) is str:
						words.append(x)
					else:
						words += x
		return words

	return extract_words(activity_type) + extract_words(place_type) + extract_words(things_to_do) + \
		   extract_words(places_to_visit) + extract_words(accommodation) + extract_words(things_to_do_p1) + \
		   extract_words(places_to_visit_p1) + exclude_words


def get_sub_destination(snippet, alt_words, h3_words, h2_words, exclude_words):
	assert type(alt_words) is list
	assert type(h3_words) is list
	assert type(h2_words) is list
	assert type(exclude_words) is list

	if snippet['h3_tag']:
		sub_dest = (set(alt_words) & set(h3_words)) - set(exclude_words)
		return ' '.join(get_ordered_words(alt_words, sub_dest))
	else:
		sub_dest = (set(alt_words) & set(h2_words)) - set(exclude_words)
		return ' '.join(get_ordered_words(alt_words, sub_dest))


def get_snippet_content_words(content, exclude_words):
	content = re.sub(r'<.*?>|{.*?}|[0-9]+?\s?[-.]?', '', content)
	words = get_words_from_text(content)
	return set(words) - set(exclude_words)


def get_snippet_title_words(title):
	words = get_tokenized_words(title)
	return [x for x in words if not is_punc(x)]


def get_sub_destination_v2(snippet, exclude_words):
	content = get_snippet_content_words(snippet['snippet_text'], exclude_words)
	title = get_snippet_title_words(snippet['blog_post_title'])
	intersection = list(content & set(title))

	if len(intersection) == 1:
		index = title.index(intersection[0])
		try:
			if not is_stop_word(title[index+1]):
				return ' '.join(title[index : index+2])
		except IndexError:
			pass
		return intersection[0]
	elif len(intersection) == 2:
		index_0 = title.index(intersection[0])
		index_1 = title.index(intersection[1])
		if abs(index_0 - index_1) <= 2:
			if index_0 > index_1:
				return ' '.join(title[index_1 : index_0+1])
			else:
				return ' '.join(title[index_0 : index_1+1])

	return ''


def process_tags(tags):
	_tags = []
	for tag in tags:
		_tags += tag.lower().split()
	return _tags


def get_main_destination(h1_words, tags, exclude_words):
	processed_tags = process_tags(tags)
	main_dest = (set(h1_words) & set(processed_tags)) - set(exclude_words)
	return ' '.join(get_ordered_words(h1_words, main_dest))


def get_snippet_location(snippet, exclude_words, tags, activity_type, place_type, things_to_do, places_to_visit,
						 accommodation, things_to_do_p1, places_to_visit_p1):
	alt_text = get_image_alt(snippet['snippet_image'])
	if alt_text:
		alt_text = clean_alt_text(alt_text)
		alt_words = get_words_from_text(alt_text)
		h3_tag = snippet['h3_tag'].replace("-"," ")
		h3_words = get_words_from_text(h3_tag)
		h2_tag = snippet['h2_tag'].replace("-"," ")
		h2_words = get_words_from_text(h2_tag)
		h1_tag = snippet['h1_tag'].replace("-"," ")
		h1_words = get_words_from_text(h1_tag)

		_exclude_words = get_words_to_exclude(activity_type, place_type, things_to_do, places_to_visit, accommodation,
											  things_to_do_p1, places_to_visit_p1, exclude_words)

		sub_dest = get_sub_destination(snippet, alt_words, h3_words, h2_words, _exclude_words) or get_sub_destination_v2(snippet, _exclude_words)
		main_dest = get_main_destination(h1_words, tags, _exclude_words)
		return [sub_dest, main_dest]
	return ['', '']
