from configparser import ConfigParser


def read_config(file_name='/home/ttlaptop0721/Projects/Python/snippet_tagging/codebase/config.ini'):
	"""
	reads the config file and return the dict like config object
	:param file_name: file name of the config file
	:return: dict like config object
	"""
	config = ConfigParser()
	config.read(file_name)
	return config


config = read_config()


def get_config(section, key):
	"""
	returns the config value for key in section.
	:param section: section name
	:param key: key name
	:return: string value
	"""
	try:
		return config[section][key]
	except KeyError:
		raise KeyError("Config Error: config key '{0}' not found in section '{1}'.".format(key, section))
