import random
import itertools
import copy
import json
import re
import logging

from collections import OrderedDict
from typing import Optional, Dict

import numpy as np

from cotk.dataloader import LanguageProcessing, MultiTurnDialog
from cotk.dataloader import Vocab, GeneralVocab, Tokenizer, SimpleTokenizer
from cotk.dataloader import Field, SentenceDefault, Sentence, Session, SessionDefault
from cotk.dataloader.field import _FieldContent
from cotk.dataloader import FieldContext, VocabContext
from cotk.file_utils import get_resource_file_path

class FakeDataLoader(LanguageProcessing):
	def __init__(self):

		self.file_id = './tests/dataloader/dummy_languageprocessing'
		self.file_path = get_resource_file_path(self.file_id)

		all_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', \
						     'what', 'how', 'here', 'do', 'as', 'can', 'to']
		set_names = ["train", "dev", "test"]
		vocab = GeneralVocab.from_predefined(all_vocab_list, 8)
		toker = SimpleTokenizer('space', ['<pad>', '<unk>', '<go>', '<eos>'])
		sent = SentenceDefault(toker, vocab, convert_to_lower_letter=True)
		fields = {set_name: [('sent', sent)] for set_name in set_names}

		with FieldContext.set_parameters(vocab=GeneralVocab.from_predefined(all_vocab_list, 8), weak=True) as field_context:

			fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]] = {}
			self.fields: "OrderedDict[str, OrderedDictType[str, Field]]" = {}
			if isinstance(fields, OrderedDict):
				fields = {set_name: fields for set_name in ["train", "dev", "test"]}
			if isinstance(fields, dict):
				for set_name, fields_in_one_set in fields.items():
					one_fields, one_fieldcontents = self._fill_field_and_create_content(set_name, fields_in_one_set)
					self.fields[set_name] = one_fields
					fieldcontents[set_name] = one_fieldcontents
			else:
				raise TypeError("Unknown type for fields")

			self._load_data(fieldcontents)

			self.vocabs = self._collect_vocabs_from_fields(self.fields)
			# self.default_vocab_id = 0 if len(self.vocabs) == 1 else None
			self.tokenizers = self._collect_tokenizers_from_fields(self.fields)
			# self.default_tokenizer_id = 0 if len(self.tokenizers) == 1 else None
			self.default_field_set_name: Optional[str] = None
			self.default_field_name: Optional[str] = None
			self._build_vocabs()

			self._setting_hash = self._create_setting_hash()
			self._vocab_hash = self._create_vocab_hash()
			self.data = self._get_data(fieldcontents)
			self._raw_data_hash, self._data_hash = self._create_data_hash(fieldcontents)
			self.index, self.batch_id, self.batch_size = self._init_batch(fieldcontents)

		self.set_default_field("train", "sent")

	# Generate a sentence
	def get_sen(self, max_len, len, gen=False, pad=True, all_vocab=False):
		sen = []
		for i in range(len):
			if all_vocab:
				vocab = random.randrange(self.eos_id, self.all_vocab_size)
			else:
				vocab = random.randrange(self.eos_id, self.frequent_vocab_size)
			if vocab == self.eos_id:
				vocab = self.unk_id
			# consider unk
			sen.append(vocab)
		if not gen:
			sen[0] = self.go_id
		sen[len - 1] = self.eos_id
		if pad:
			for i in range(max_len - len):
				sen.append(self.pad_id)
		return sen


	def get_data(self, reference_key=None, reference_len_key=None, gen_prob_key=None, gen_key=None, \
				 post_key=None, \
				 to_list=False, \
				 pad=True, gen_prob_check='no_check', \
				 gen_len='random', ref_len='random', \
				 ref_vocab='all_vocab', gen_vocab='all_vocab', gen_prob_vocab='all_vocab', \
				 resp_len='>=2', batch=5, max_len=10, reference_num=1):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			post_key: [] \
		}

		for i in range(batch):
			data[reference_key].append([])
			data[reference_len_key].append([])
			for _ in range(reference_num):
				if resp_len == '<2':
					ref_nowlen = 1
				elif ref_len == "random":
					ref_nowlen = random.randrange(2, 5)
				elif ref_len == "non-empty":
					ref_nowlen = 8
				elif ref_len == 'empty':
					ref_nowlen = 2
				
				sen = self.get_sen(max_len, ref_nowlen, pad=pad, all_vocab=ref_vocab=='all_vocab')
				if reference_num == 1:
					data[reference_key][-1] = sen
					data[reference_len_key][-1] = ref_nowlen
				else:
					data[reference_key][-1].append(sen)
					data[reference_len_key][-1].append(ref_nowlen)
	
			data[post_key].append(self.get_sen(max_len, ref_nowlen, pad=pad))

			if gen_len == "random":
				gen_nowlen = random.randrange(1, 4) if i > 2 else 3 # for BLEU not empty
			elif gen_len == "non-empty":
				gen_nowlen = 7
			elif gen_len == "empty":
				gen_nowlen = 1
			data[gen_key].append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad, \
											  all_vocab=gen_vocab=='all_vocab'))

			gen_prob = []
			for j in range(ref_nowlen - 1):
				vocab_prob = []
				if gen_prob_vocab == 'all_vocab':
					vocab_nowsize = self.all_vocab_size
				else:
					vocab_nowsize = self.frequent_vocab_size

				for k in range(vocab_nowsize):
					vocab_prob.append(random.random())
				vocab_prob /= np.sum(vocab_prob)
				if gen_prob_check != "random_check":
					vocab_prob = np.log(vocab_prob)
				gen_prob.append(list(vocab_prob))
			data[gen_prob_key].append(gen_prob)

		if gen_prob_check == "full_check":
			data[gen_prob_key][-1][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

class FakeMultiDataloader(MultiTurnDialog):
	def __init__(self):

		self.file_id = "./tests/dataloader/dummy_ubuntucorpus#Ubuntu"
		self.file_path = get_resource_file_path(self.file_id)

		all_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', \
						   'what', 'how', 'here', 'do', 'as', 'can', 'to']
		set_names = ["train", "dev", "test"]
		vocab = GeneralVocab.from_predefined(all_vocab_list, 8)
		toker = SimpleTokenizer('space', ['<pad>', '<unk>', '<go>', '<eos>'])
		sent = SessionDefault(toker, vocab, convert_to_lower_letter=True)
		fields = {set_name: [('session', sent)] for set_name in set_names}

		with FieldContext.set_parameters(vocab=GeneralVocab.from_predefined(all_vocab_list, 8),
										 weak=True) as field_context:

			fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]] = {}
			self.fields: Dict[str, OrderedDictType[str, Field]] = {}
			if isinstance(fields, OrderedDict):
				fields = {set_name: fields for set_name in ["train", "dev", "test"]}
			if isinstance(fields, dict):
				for set_name, fields_in_one_set in fields.items():
					one_fields, one_fieldcontents = self._fill_field_and_create_content(set_name, fields_in_one_set)
					self.fields[set_name] = one_fields
					fieldcontents[set_name] = one_fieldcontents
			else:
				raise TypeError("Unknown type for fields")

			self._load_data(fieldcontents)

			self.vocabs = self._collect_vocabs_from_fields(self.fields)
			# self.default_vocab_id = 0 if len(self.vocabs) == 1 else None
			self.tokenizers = self._collect_tokenizers_from_fields(self.fields)
			# self.default_tokenizer_id = 0 if len(self.tokenizers) == 1 else None
			self.default_field_set_name: Optional[str] = None
			self.default_field_name: Optional[str] = None
			self._build_vocabs()

			self._setting_hash = self._create_setting_hash()
			self._vocab_hash = self._create_vocab_hash()
			self.data = self._get_data(fieldcontents)
			self._raw_data_hash, self._data_hash = self._create_data_hash(fieldcontents)
			self.index, self.batch_id, self.batch_size = self._init_batch(fieldcontents)

		self.set_default_field("train", "session")


	def get_sen(self, max_len, len, gen=False, pad=True, all_vocab=False):
		return FakeDataLoader.get_sen(self, max_len, len, gen, pad, all_vocab)

	def get_data(self, reference_key=None, reference_len_key=None, turn_len_key=None, gen_prob_key=None, gen_key=None, \
				 context_key=None, \
				 to_list=False, \
				 pad=True, gen_prob_check='no_check', \
				 gen_len='random', ref_len='random', \
				 ref_vocab='all_vocab', gen_vocab='all_vocab', gen_prob_vocab='all_vocab', \
				 resp_len='>=2', batch=5, max_len=10, max_turn=5, test_prec_rec=False):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			turn_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			context_key: [] \
		}

		for i in range(batch):
			if test_prec_rec:
				turn_length = 3
			else:
				turn_length = random.randrange(1, max_turn+1)
			turn_reference = []
			turn_reference_len = []
			turn_gen_prob = []
			turn_gen = []
			turn_context = []

			for j in range(turn_length):
				if resp_len == '<2':
					ref_nowlen = 1
				elif ref_len == "random":
					ref_nowlen = random.randrange(3 if test_prec_rec else 2, 5)
				elif ref_len == "non-empty":
					ref_nowlen = 8
				elif ref_len == 'empty':
					ref_nowlen = 2
				turn_reference.append(self.get_sen(max_len, ref_nowlen, pad=pad, all_vocab=ref_vocab=='all_vocab'))
				turn_reference_len.append(ref_nowlen)

				turn_context.append(self.get_sen(max_len, ref_nowlen, pad=pad, all_vocab=ref_vocab=='all_vocab'))
				if gen_len == "random":
					gen_nowlen = random.randrange(1, 4) if i != 0 and not test_prec_rec else 3 # for BLEU not empty
				elif gen_len == "non-empty":
					gen_nowlen = 7
				elif gen_len == "empty":
					gen_nowlen = 1
				turn_gen.append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad, all_vocab=gen_vocab=='all_vocab'))

				gen_prob = []
				for k in range(max_len - 1 if pad else ref_nowlen - 1):
					vocab_prob = []
					if gen_prob_vocab == 'all_vocab':
						vocab_nowsize = self.all_vocab_size
					else:
						vocab_nowsize = self.frequent_vocab_size

					for l in range(vocab_nowsize):
						vocab_prob.append(random.random())
					vocab_prob /= np.sum(vocab_prob)
					if gen_prob_check != "random_check":
						vocab_prob = np.log(vocab_prob)
					gen_prob.append(list(vocab_prob))
				turn_gen_prob.append(gen_prob)

			data[reference_key].append(turn_reference)
			data[reference_len_key].append(turn_reference_len)
			data[turn_len_key].append(turn_length)
			data[gen_prob_key].append(turn_gen_prob)
			data[gen_key].append(turn_gen)
			data[context_key].append(turn_context)

		if gen_prob_check == "full_check":
			data[gen_prob_key][-1][-1][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

test_dataloader = ['dataloader', 'field']

test_argument =  [ 'default',   'custom']

test_shape =     [     'pad',      'jag',      'pad',      'jag']
test_type =      [   'array',    'array',     'list',     'list']

test_batch_len = [   'equal',  'unequal']
test_turn_len =  [   'equal',  'unequal']

test_check =     ['no_check', 'random_check', 'full_check']

test_gen_len =   [  'random', 'non-empty',   'empty']
test_ref_len =   [  'random', 'non-empty',   'empty']

test_ref_vocab =      ['valid_vocab', 'all_vocab']
test_gen_vocab =      ['valid_vocab', 'all_vocab']
test_gen_prob_vocab = ['valid_vocab', 'all_vocab']

test_resp_len = ['>=2', '<2']
test_include_invalid = [False, True]
#test_ngram = [1, 2, 3, 4, 5, 6]
test_ngram = [1, 2, 3, 4]

test_emb_mode = ['avg', 'extrema', 'sum']
test_emb_type = ['dict', 'list']
test_emb_len = ['equal', 'unequal']

test_hash_data = ['has_key', 'no_key']

## test_batch_len: len(ref) == len(gen)?
## test_turn_len: len(single_batch(ref)) == len(single_batch(gen))?
## test_gen_len: 'empty' means all length are 1 (eos), 'non-empty' means all length are > 1, 'random' means length are random
## test_ref_len: 'empty' means all length are 2 (eos), 'non-empty' means all length are > 2, 'both' means length are random

def same_data(A, B, exact_equal=True):
	if type(A) != type(B):
		return False
	if isinstance(A, str):
		return A == B
	try:
		if len(A) != len(B):
			return False
	except TypeError:
		if not exact_equal and isinstance(A, float):
			return np.isclose(A, B)
		return A == B
	for i, x in enumerate(A):
		if not same_data(x, B[i]):
			return False
	return True

def same_dict(A, B, exact_equal=True):
	if A.keys() != B.keys():
		return False
	for x in A.keys():
		if not same_data(A[x], B[x], exact_equal):
			return False
	return True

def generate_testcase(*args):
	args = [(list(p), mode) for p, mode in args]
	default = []
	for p, _ in args:
		default.extend(p[0])
	yield tuple(default)
	# add
	i = 0
	for p, mode in args:
		if mode == "add":
			for k in p[1:]:
				yield tuple(default[:i] + list(k) + default[i+len(p[0]):])
		i += len(p[0])

	# multi
	res = []
	for i, (p, mode) in enumerate(args):
		if mode == "add":
			res.append(p[:1])
		else:
			res.append(p)
	for p in itertools.product(*res):
		yield tuple(itertools.chain(*p))

def replace_unk( _input, _target=-1):
	output = []
	for _list in _input:
		_output = []
		for ele in _list:
			_output.append(_target if ele == 1 else ele)
		output.append(_output)
	return output

def shuffle_instances(data, key_list):
	indices = list(range(len(data[key_list[0]])))
	np.random.shuffle(indices)
	data_shuffle = copy.deepcopy(data)
	for key in key_list:
		if isinstance(data_shuffle[key], list):
			data_shuffle[key] = [data_shuffle[key][idx] for idx in indices]
		else:
			data_shuffle[key] = data_shuffle[key][indices]
	return data_shuffle

def split_batch(data, key_list, \
				less_pad=False, to_list=False, reference_key=None, reference_is_3D=False):
	'''Split one batch into two

	Arguments:
		* less_pad (bool): if `True`, the length of padding in the two batches are different
						   if `False`, the length of padding in the two batches are the same

	'''
	batches = []
	for idx in range(2):
		tmp = {}
		for key in key_list:
			if idx == 0:
				tmp[key] = data[key][:1]
			else:
				tmp[key] = data[key][1:]
		batches.append(tmp)
	if less_pad:
		if not reference_is_3D:
			if to_list:
				batches[0][reference_key][0] = batches[0][reference_key][0][:-1]
			else:
				batches[0][reference_key] = np.array(batches[0][reference_key].tolist())[:, :-1]
		else:
			if to_list:
				tmp = []
				for lst in batches[0][reference_key][0]:
					tmp.append(lst[:-1])
				batches[0][reference_key][0] = tmp
			else:
				batches[0][reference_key] = np.array(batches[0][reference_key].tolist())[:, :, :-1]
	return batches

def generate_unequal_data(data, key_list, pad_id, reference_key, \
						  reference_len_key=None, reference_is_3D=False):
	res = []
	for unequal_type in ['less_inst', 'less_word', 'change_word', 'shuffle_words']:
		data_unequal = copy.deepcopy(data)
		if unequal_type == 'less_inst':
			for key in key_list:
				data_unequal[key] = data_unequal[key][1:]
		elif unequal_type == 'less_word':
			if reference_len_key is None:
				continue
			if reference_is_3D:
				data_unequal[reference_len_key][0][0] -= 2
			else:
				data_unequal[reference_len_key][0] -= 2
		elif unequal_type == 'change_word':
			if reference_is_3D:
				data_unequal[reference_key][0][0][1] = pad_id
			else:
				data_unequal[reference_key][0][1] = pad_id
		else:
			if reference_is_3D:
				np.random.shuffle(data_unequal[reference_key][0][0])
			else:
				np.random.shuffle(data_unequal[reference_key][0])
		res.append(data_unequal)
	return res

def version_test(metric_class, dataloader=None):
	name = metric_class._name
	version = metric_class._version
	filename = './tests/metric/version_test_data/{}_v{}.jsonl'.format(name, version)
	if isinstance(dataloader, FakeMultiDataloader):
		tmp_file_id = "./tests/dataloader/dummy_ubuntucorpus#Ubuntu"
	else:
		tmp_file_id = './tests/dataloader/dummy_languageprocessing'
	with open(filename, "r") as file:
		for line in file:
			data = json.loads(line)
			if dataloader:
				tmp_set_names = ["train", "dev", "test"]
				tmp_vocab = GeneralVocab.from_predefined(data['init']['dataloader']['all_vocab_list'], \
														 	data['init']['dataloader']['valid_vocab_len'])
				tmp_toker = SimpleTokenizer('space', ['<pad>', '<unk>', '<go>', '<eos>'])
				if isinstance(dataloader, FakeMultiDataloader):
					tmp_sent = SessionDefault(tmp_toker, tmp_vocab, convert_to_lower_letter=True)
					tmp_fields = {set_name: [('session', tmp_sent)] for set_name in tmp_set_names}
				else:
					tmp_sent = SentenceDefault(tmp_toker, tmp_vocab, convert_to_lower_letter=True)
					tmp_fields = {set_name: [('sent', tmp_sent)] for set_name in tmp_set_names}

				tmp_dataloader = dataloader.simple_create(tmp_file_id, tmp_fields)
				if isinstance(dataloader, FakeMultiDataloader):
					tmp_dataloader.set_default_field("train", "session")
				else:
					tmp_dataloader.set_default_field("train", "sent")
				data['init']['dataloader'] = tmp_dataloader

			metric = metric_class(**data['init'])
			for batch in data['forward']:
				metric.forward(**batch)
			res = metric.close()
			for key, val in res.items():
				if isinstance(val, float) or re.match(r"<class 'numpy\.float\d*'>", str(type(val))):
					res[key] = float(val)
				elif isinstance(val, int) or re.match(r"<class 'numpy\.int\d*'>", str(type(val))):
					res[key] = int(val)
			print(res)
			print(data['output'])
			assert same_dict(res, data['output'], exact_equal=False), "Version {} error".format(version)
			# assert metric.close() == data['output'], "Version {} error".format(version)
