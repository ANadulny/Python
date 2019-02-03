from collections import OrderedDict


class CitiesDataCleaner:
	'''
	Cleaner of cities data
	'''

	@classmethod
	def clean(cls, rows):
		for row in rows:
			# czyszczenie wspólne dla wszystkich kolumn
			clean_row = cls._clean_all_items(row)
			
			# czyszczenie wyspecjalizowane
			clean_row['towns'] = (
				cls._clean_towns_name(clean_row['towns'])
			)

			yield clean_row
		
	@staticmethod
	def _clean_all_items(row):
		clean_row = OrderedDict()
		for key, value in row.items():
			clean_value = value.strip()
			clean_value = clean_value or None
			clean_row[key] = clean_value
		return clean_row

	@staticmethod
	def _clean_towns_name(towns):
		town_name = towns.strip('.')
		return town_name 


def filter_input_data(orderedDict):
	return [x for x in orderedDict if int(x['population']) > 60000]


if __name__ == '__main__':
	pass
	# data = get_csv_lines(relative_path(__file__, in_filename))
	# cleaned_data = CitiesDataCleaner.clean(data)
	# cleaned_data = filter_input_data(cleaned_data)
	# print(len(list(cleaned_data)))
	# a = Genotype(cleaned_data)
	# a.get_city_data()
	# print(a.get_mutable())
	# print(a.create_fenotype())




