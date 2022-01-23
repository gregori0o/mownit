from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox

from nltk.tokenize import word_tokenize as tokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from scipy.sparse.linalg import svds as svd
from scipy.sparse import lil_matrix


class ResultButton (Button):
	def __init__(self, filepath, val, **kwargs):
		super (ResultButton, self).__init__(**kwargs)

		f = open(filepath, 'r')
		self.site = f.read()
		f.close()
		self.title = self.site.split('\n')[0]
		self.text = self.title + " <--- " + str (val)
		self.halign = 'center'

	def on_press (self):
		layout = BoxLayout(orientation='vertical')
		label = Label(text=self.site, valign='top', size_hint=(1, None), text_size = (self.width, None))
		label.bind(size=label.setter('text_size'))
		label.bind(texture_size=label.setter('size'))
		label.bind(size_hint_min_x=label.setter('width'))
		scroll = ScrollView()
		scroll.add_widget(label)
		layout.add_widget(scroll)
		button = Button (text="<-- BACK", size_hint=(1, .1))
		layout.add_widget(button)
		self.popup = Popup (title=self.title,
			title_align='center',
			content=layout,
			size_hint=(1, 1),
			auto_dismiss=False)
		button.bind(on_press = self.popup.dismiss)
		self.popup.open()


class ResultSearch (Screen):
	def __init__(self, question, k, app, **kwargs):
		super (ResultSearch, self).__init__(**kwargs)

		self.app = app
		
		layout = BoxLayout (orientation='vertical')

		self.buttons = []
		for i, val in self.make_search(question, k):
			self.buttons.append(ResultButton(self.app.filepaths[int(i)], val))
			layout.add_widget(self.buttons[-1])

		exit = Button (text="<--- EXIT")
		exit.bind(on_press = self.exit)
		layout.add_widget(exit)

		self.add_widget(layout)

	def make_search (self, question, k):
		vector = self.make_bag_of_word (question)
		result = np.zeros((len(self.app.vectors), 2))
		result[:, 0] = np.arange(len(self.app.vectors))
		for i, v in enumerate(self.app.vectors):
			result[i, 1] = self.cosine (vector, v)
		result = sorted(result, key=lambda x : x[1], reverse=True)
		return np.array(result)[:k, :]

	def make_bag_of_word (self, text):
		text = text.lower()
		words = tokenizer (text)
		vector = {}
		for word in words:
			word = self.app.stemmer.stem(word)
			idx = self.app.dictionary.get(word,-1)
			if idx == -1:
				continue
			vector[idx] = vector.get(idx, 0) + 1
		s = 0
		for v in vector.values ():
			s += v**2
		s = np.sqrt (s)
		if s == 0:
			return vector
		for key in vector.keys():
			vector[key] /= s
		return vector

	def cosine (self, v, u):
		l = 0
		if isinstance (u, dict):
			for key, i in v.items():
				l += (i * u.get(key, 0))
		else:
			for key, i in v.items():
				l += (i * u[key])
			l /= np.sqrt(np.sum(u**2))
		return l

	def exit (self, *args):
		self.manager.current = 'start'
		self.app.screenmanager.remove_widget(self)


class StartScreen (Screen):
	def __init__(self, app, **kwargs):
		super (StartScreen, self).__init__(**kwargs)
		self.app = app

		layout = FloatLayout ()

		title = Label (text="Wyszukiwarka Wikipedii",
			halign='center',
			font_size='40sp',
			size_hint=(.8, .2),
			pos_hint={'x': .1, 'y': 0.7})
		layout.add_widget(title)

		label1 = Label (text="Wpisz zapytanie",
			halign='center',
			font_size='20sp',
			size_hint=(.8, .1),
			pos_hint={'x': .1, 'y': 0.6})
		layout.add_widget(label1)
		self.question = TextInput (font_size='30sp',
			size_hint=(.6, .1),
			pos_hint={'x': .2, 'y': 0.5})
		layout.add_widget(self.question)

		label2 = Label (text="Wpisz liczbę żądanych wyszukiwań",
			halign='center',
			font_size='20sp',
			size_hint=(.8, .1),
			pos_hint={'x': .1, 'y': 0.4})
		layout.add_widget(label2)
		self.numbers = TextInput (text="10",
			font_size='30sp',
			size_hint=(.6, .1),
			pos_hint={'x': .2, 'y': 0.3})
		layout.add_widget(self.numbers)

		button = Button (text="SEARCH",
			font_size='20sp',
			size_hint=(.8, .15),
			pos_hint={'x': .1, 'y': .05})
		button.bind(on_press = self.search)
		layout.add_widget(button)

		button_back = Button (text="EXIT",
			font_size='20sp',
			size_hint=(.3, .1),
			pos_hint={'x': .0, 'y': .9})
		button_back.bind(on_press = self.exit)
		layout.add_widget(button_back)

		self.add_widget(layout)

	def exit (self, *args):
		self.dictionary = None
		self.vectors, self.filepaths = None, None
		self.manager.current = 'prepare'

	def search (self, *args):
		quest = self.question.text
		k = int (self.numbers.text)
		self.question.text = ""
		self.app.screenmanager.add_widget(ResultSearch(quest, k, self.app, name='result'))
		self.manager.current = 'result'


class PrepareData (Screen):
	def __init__(self, app, **kwargs):
		super (PrepareData, self).__init__(**kwargs)
		self.app = app

		layout = FloatLayout ()

		title = Label (text="Proszę wybrać w jaki sposób przygotować dane, lub skorzystaj z automatycznego wyboru",
			halign='center',
			font_size='20sp',
			size_hint=(.8, .2),
			pos_hint={'x': .1, 'y': 0.8})
		layout.add_widget(title)

		label0 = Label (text="full dict",
			halign='left',
			font_size='20sp',
			size_hint=(.4, .1),
			pos_hint={'x': .1, 'y': 0.7})
		layout.add_widget(label0)
		self.box0 = CheckBox(size_hint=(.2, .1),
			pos_hint={'x': .6, 'y': 0.7})
		layout.add_widget(self.box0)

		label1 = Label (text="IDFT",
			halign='left',
			font_size='20sp',
			size_hint=(.4, .1),
			pos_hint={'x': .1, 'y': 0.6})
		layout.add_widget(label1)
		self.box1 = CheckBox(size_hint=(.2, .1),
			pos_hint={'x': .6, 'y': 0.6})
		layout.add_widget(self.box1)

		label2 = Label (text="SVD",
			halign='left',
			font_size='20sp',
			size_hint=(.4, .1),
			pos_hint={'x': .1, 'y': 0.5})
		layout.add_widget(label2)
		self.box2 = CheckBox(size_hint=(.2, .1),
			pos_hint={'x': .6, 'y': 0.5})
		layout.add_widget(self.box2)

		label3 = Label (text="k in SVD",
			halign='left',
			font_size='20sp',
			size_hint=(.4, .1),
			pos_hint={'x': .1, 'y': 0.4})
		layout.add_widget(label3)
		self.box3 = TextInput (text="120",
			halign='center',
			font_size='30sp',
			size_hint=(.2, .1),
			pos_hint={'x': .6, 'y': 0.4})
		layout.add_widget(self.box3)

		default_button = Button (text="DEFAULT",
			font_size='20sp',
			size_hint=(.3, .2),
			pos_hint={'x': .1, 'y': .1})
		default_button.bind(on_press = self.default)
		layout.add_widget(default_button)

		prepare_button = Button (text="PREPARE",
			font_size='20sp',
			size_hint=(.3, .2),
			pos_hint={'x': .6, 'y': .1})
		prepare_button.bind(on_press = self.prepare)
		layout.add_widget(prepare_button)

		self.add_widget(layout)

	def default (self, *args):
		self.app.dictionary = self.read_dictionary ()
		self.app.vectors, self.app.filepaths = self.read_data ()
		self.prepare_vectors()
		self.manager.current = 'start'

	def prepare (self, *args):
		if self.box0.active:
			self.app.dictionary = self.read_dictionary ("dict_full.txt")
			self.app.vectors, self.app.filepaths = self.read_data ("data_full.txt")
		else:
			self.app.dictionary = self.read_dictionary ()
			self.app.vectors, self.app.filepaths = self.read_data ()

		if self.box2.active:
			k = int(self.box3.text)
		else:
			k = -1
		self.prepare_vectors(self.box1.active, k)
		self.manager.current = 'start'

	def prepare_vectors (self, IDF=True, k=120):
		if IDF:
			self.app.vectors = self.IDF_func (self.app.vectors)
		self.app.vectors = self.normalise_vectors (self.app.vectors)
		if k > 0:
			self.app.vectors = self.reduce_noise (self.app.vectors, k, len(self.app.dictionary))

	def normalise (self, vector):
		s = 0
		for v in vector.values():
			s += v**2
		s = np.sqrt(s)
		if s == 0:
			return vector
		for key in vector.keys():
			t = vector[key]
			vector[key] = t / s
		return vector

	def normalise_vectors (self, vectors):
		for i in range (len (vectors)):
			vectors[i] = self.normalise(vectors[i])
		return vectors

	def IDF_func (self, vectors):
		n = len(self.app.dictionary)
		N = len (vectors)
		IDF = np.zeros(n)
		for vector in vectors:
			for key in vector.keys():
				IDF[key] += 1
		IDF = np.log(N / IDF)
		for i in range (N):
			for key in vectors[i].keys():
				vectors[i][key] *= IDF [key]
		return vectors

	def reduce_noise (self, vectors, k, size):
		l = len (vectors)
		matrix = lil_matrix((l, size), dtype=np.float64)
		for i, vector in enumerate(vectors):
			for j, val in vector.items():
				matrix[i, j] = val
		u, s, vh = svd(matrix, k=k)
		del (matrix)
		u = u.astype(np.float32)
		s = s.astype(np.float32)
		vh = vh.astype(np.float32)
		new_matrix = u @ np.diag(s) @ vh
		return new_matrix

	def read_data (self, filepath = 'raw_data.txt'):
		arr = []
		filepaths = []
		with open(filepath, 'r') as f:
			for line in f:
				vector = {}
				if len(line) < 3:
					continue
				string = line.split('$')[2][1:][:-2]
				if len(string) < 3:
					continue
				if ',' in string:
					string = string.split(', ')
				elif ' ' in string and ':' not in string:
					string = string.split()
				else:
					string = [string]
				for i, item in enumerate(string):
					if ':' in item:
						key, val = item.split(': ')
					else:
						key, val = i, item
					vector[int(key)] = float (val)
				filepaths.append (line.split('$')[1])
				arr.append(vector)
		return arr, filepaths

	def read_dictionary (self, filepath = 'dict.txt'):
		result = {}
		f = open (filepath, 'r')
		text = f.read()
		f.close()
		text = text[1:][:-1]
		text = text.split(', ')
		for item in text:
			key, value = item.split(': ')
			if "'" in key:
				key = key.split("'")[1]
			result[key] = int(value)
		return result


class MainApp(App):
	def build(self):
		self.dictionary = None
		self.vectors, self.filepaths = None, None
		self.stemmer = SnowballStemmer("english")

		self.screenmanager = ScreenManager(transition=FadeTransition())
		self.screenmanager.add_widget(PrepareData(self, name='prepare'))
		self.screenmanager.add_widget(StartScreen(self, name='start'))
		return self.screenmanager


if __name__ == "__main__":
    app = MainApp()
    app.run()