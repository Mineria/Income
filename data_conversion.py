"""
Bunch of functions to convert a given string
into their numerical representation
"""
def workclass(clase):
	if clase == 'Private':
		valor = 1
	elif clase == 'Self-emp-not-inc':
		valor = 2
	elif clase == 'Self-emp-inc':
		valor = 3
	elif clase == 'Never-worked':
		valor = 4
	elif clase == 'Federal-gov':
		valor = 5
	elif clase == 'Local-gov':
		valor = 6
	elif clase == 'State-gov':
		valor = 7
	else:
		valor = 0
	return valor

def education(eduacion):
	if eduacion == 'Doctorate':
		valor = 1
	elif eduacion == 'Masters':
		valor = 2
	elif eduacion == 'Bachelors':
		valor = 3
	elif eduacion == 'Some-college':
		valor = 4
	elif eduacion == 'Prof-school':
		valor = 5
	elif eduacion == 'HS-grad':
		valor = 6
	elif eduacion == '7th-8th':
		valor = 7
	elif (eduacion == 'Assoc-acdm') or ( eduacion == 'Assoc-voc'):
		valor = 8
	elif (eduacion == '1st-4th') or ( eduacion == '5th-6t'):
		valor = 9
	elif (eduacion == '11th') or ( eduacion == '9th') or ( eduacion == '12th') or ( eduacion == '10th'):
		valor = 10
	else:
		valor = 0
	return valor

def marital_status(estatus):
	if estatus == 'Married-civ-spouse':
		valor = 1
	elif estatus == 'Divorced':
		valor = 2
	elif estatus == 'Never-married':
		valor = 3
	elif estatus == 'Separated':
		valor = 4
	elif estatus == 'Widowed':
		valor = 5
	elif estatus == 'Married-spouse-absent':
		valor = 6
	else:
		valor = 0
	return valor

def occupation(ocupacion):
	if ocupacion == 'Tech-support':
		valor = 1
	elif ocupacion == 'Craft-repair':
		valor = 2
	elif ocupacion == 'Sales':
		valor = 3
	elif ocupacion == 'Exec-managerial':
		valor = 4
	elif ocupacion == 'Prof-specialty':
		valor = 5
	elif ocupacion == 'Handlers-cleaners':
		valor = 6
	elif ocupacion == 'Machine-op-inspct':
		valor = 7
	elif ocupacion == 'Adm-clerical':
		valor = 8
	elif ocupacion == 'Farming-fishing':
		valor = 9
	elif ocupacion == 'Transport-moving':
		valor = 10
	elif ocupacion == 'Priv-house-serv':
		valor = 11
	elif ocupacion == 'Protective-serv':
		valor = 12
	elif ocupacion == 'Armed-Forces':
		valor = 13
	else:
		valor = 0
	return valor

def relationship(relacion):
	if relacion == 'Wife':
		valor = 1
	elif relacion == 'Own-child':
		valor = 2
	elif relacion == 'Husband':
		valor = 3
	elif relacion == 'Not-in-family':
		valor = 4
	elif relacion == 'Unmarried':
		valor = 5
	else:
		valor = 0
	return valor

def race(raza):
	if raza == 'White':
		valor = 1
	elif raza == 'Asian-Pac-Islander':
		valor = 2
	elif raza == 'Amer-Indian-Eskimo':
		valor = 3
	elif raza == 'Black':
		valor = 4
	else:
		valor = 0
	return valor

def sex(sexo):
	if sexo == 'Female':
		valor = 1
	else:
		valor = 0
	return valor

def native_country(paisNativo):
	if paisNativo == 'United-States':
		valor = 1
	elif (paisNativo == 'England') or ( paisNativo == 'Canada') or ( paisNativo == 'Germany') or ( paisNativo == 'Japan'):
		valor = 2
	elif (paisNativo == 'Greece') or ( paisNativo == 'Italy') or ( paisNativo == 'Poland') or ( paisNativo == 'Portugal') or ( paisNativo == 'Ireland') or ( paisNativo == 'France') or ( paisNativo == 'Hungary') or ( paisNativo == 'Scotland') or ( paisNativo == 'Yugoslavia') or ( paisNativo == 'Holand-Netherlands'):
		valor = 3
	elif (paisNativo == 'Puerto-Rico') or ( paisNativo == 'Outlying-US(Guam-USVI-etc)') or ( paisNativo == 'South') or ( paisNativo == 'Cuba') or ( paisNativo == 'Honduras') or ( paisNativo == 'Mexico') or ( paisNativo == ' Dominican-Republic') or ( paisNativo == 'Ecuador') or ( paisNativo == 'Columbia') or ( paisNativo == 'Guatemala') or ( paisNativo == 'Nicaragua') or ( paisNativo == 'El-Salvador') or ( paisNativo == 'Trinadad&Tobago') or ( paisNativo == 'Peru'):
		valor = 4
	elif (paisNativo == 'Cambodia') or ( paisNativo == 'India') or ( paisNativo == 'China') or ( paisNativo == 'Iran') or ( paisNativo == 'Vietnam') or ( paisNativo == 'Philippines') or ( paisNativo == 'Laos') or ( paisNativo == 'Taiwan') or ( paisNativo == 'Thailand') or ( paisNativo == 'Hong')  :
		valor = 5
	else:
		valor = 0
	return valor

def income(income):
	value = 0
	if income == "<=50K":
		value = 0
	elif income == ">50K":
		value = 1
	return value
