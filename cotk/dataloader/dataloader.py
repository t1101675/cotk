'''A module for dataloader'''
import random
from typing import Optional, Any, Union, Sequence, Dict, Tuple, Iterable, List
from collections import Counter, OrderedDict
from itertools import chain
import logging
from hashlib import sha256

import numpy as np

from .._utils import trim_before_target
from .._utils.unordered_hash import UnorderedSha256, dumps
from .._utils.metaclass import DocStringInheritor, LoadClassInterface, copy_func, copy_property
from .._utils.typehint import OrderedDictType
from ..file_utils import get_resource_file_path
from .tokenizer import Tokenizer
from .field import Field, SentenceDefault, _FieldContent, Sentence
from .vocab import Vocab, GeneralVocab
from .context import FieldContext, VocabContext

class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''

class LanguageProcessing(Dataloader):
	r"""Bases: :class:'.dataloader.Dataloader'

	Base class for all language processing tasks. This is an abstract class.
	During the initialization of the dataloader, :class:`Vocab` or :class:`Field` may be created.
	To specifiy the parameters of these created object, please use :class:`VocabContext`
	and :class:`FieldContext`, or just use :meth:`.simple_create`.
	See the examples for how to create a dataloader. #TODO: write the example

	Arguments:{ARGUMENTS}
	"""

	ARGUMENTS = r"""
			file_id (str): A string indicating the dataset. It can be local path ("./data"), a resource name
				(resources://dataset), or an url (http://test.com/dataset.zip).
			fields (OrderedDict[str, Union[str, Field]], Dict[str, OrderedDict[str, Union[str, Field]]]):
				If ``OrderDict``, it describes the data format of the "train", "dev", "test" set.
				If ``Dict``, ``fields[key]`` describes the data format of the set named ``key``.
				The data format is an ordered dictionary, where ``key`` is the name of a field,
				``value`` is either a string indicating a Field or a :class:`Field` object.
				See the examples for how to specify the data format. #TODO: write the example"""

	def __init__(self, file_id: str, \
				 fields: Union[OrderedDictType[str, Union[str, Field]],\
					 		   Dict[str, OrderedDictType[str, Union[str, Field]]]], \
				 ):
		self.file_id = file_id
		self.file_path = get_resource_file_path(file_id)

		field_context = FieldContext.set_parameters(vocab=GeneralVocab(), weak=True)

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

		field_context.close()

	@staticmethod
	def simple_create(file_id: str, \
				fields: Union[OrderedDictType[str, Union[str, Field]],\
					 		   Dict[str, OrderedDictType[str, Union[str, Field]]]], \
				*,\
				tokenizer: Union[Tokenizer, str, None] = None, \
				vocab: Optional[Vocab] = None, \
				vocab_from: Optional[Dict[str, str]] = None, \
				max_sent_length: Optional[int] = None, \
				max_turn_length: Optional[int] = None, \
				convert_to_lower_letter: Optional[bool] = None, \
				min_frequent_vocab_times: Optional[int] = None, \
				min_rare_vocab_times: Optional[int] = None, \
				special_tokens_mapping: Optional[OrderedDictType[str, str]] = None, \
				special_appeared_in_data: Optional[bool] = None) -> "LanguageProcessing":
		'''A simple way to create a dataloader. Instead of using :class:`VocabContext`
		and :class:`FieldContext`, specifying all the possible parameters here.

		Arguments:{ARGUMENTS}
		TODO: more arguments from VocabContext, FieldContext
		'''
		with VocabContext.set_parameters(\
				min_frequent_vocab_times=min_frequent_vocab_times,\
				min_rare_vocab_times=min_rare_vocab_times, \
				special_tokens_mapping=special_tokens_mapping, \
				special_appeared_in_data=special_appeared_in_data):
			with FieldContext.set_parameters(\
					tokenizer=tokenizer, \
					vocab=vocab, \
					vocab_from=vocab_from, \
					max_sent_length=max_sent_length, \
					max_turn_length=max_turn_length, \
					convert_to_lower_letter=convert_to_lower_letter):
				with FieldContext.set_parameters(tokenizer="space", weak=True):
					return LanguageProcessing(file_id, fields)

	def _load_data(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]):
		'''Load data from file.

		Arguments:
			fieldcontents (Dict[str, OrderedDictType[str, _FieldContent]]): fieldcontents for each set
		'''
		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			with open("%s/%s.txt" % (self.file_path, set_name), encoding='utf-8') as f_file:
				line_cnt = 0
				file_iterator = iter(f_file)
				while True:
					try:
						for _, fieldcontent in fieldcontents_in_one_set.items():
							line_add = fieldcontent.read_next(file_iterator)
							if line_add == 0:
								while True:
									if next(file_iterator):
										raise RuntimeError("the file %s corrupted at line %d" % (set_name, line_cnt))
							line_cnt += line_add
					except StopIteration:
						break

			sample_nums = [fieldcontent.get_data_number() for _, fieldcontent in fieldcontents_in_one_set]
			if not all([sample_num == sample_nums[0] for sample_num in sample_nums]):
				raise RuntimeError("the file %s corrupted at end of the file")

		for _, fieldcontents_in_one_set in fieldcontents.items():
			for _, fieldcontent in fieldcontents_in_one_set.items():
				fieldcontent.process_before_vocab()

	def _init_batch(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]) -> \
			Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, Optional[int]]]:
		'''Initialize the batches. Return a tuple contains
		``index``, ``batch_id``, ``batch_size`` for each set.

		Arguments:
			fieldcontents (Dict[str, OrderedDictType[str, _FieldContent]]): fieldcontents for each set.
		'''
		index: Dict[str, List[int]] = {}
		batch_id: Dict[str, int] = {}
		batch_size: Dict[str, Optional[int]] = {}

		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			sample_nums = [fieldcontent.get_data_number() \
					for _, fieldcontent in fieldcontents_in_one_set.items()]
			batch_id[set_name] = 0
			batch_size[set_name] = None
			index[set_name] = list(range(sample_nums[0]))

		return index, batch_id, batch_size

	def _get_data(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]) -> \
			Dict[str, Dict[str, Any]]:
		'''Get the data from fieldcontents.

		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, _FieldContent]]): fieldcontents for each set.
		'''
		data: Dict[str, Dict[str, Any]] = {}
		for set_name, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			data[set_name] = {}
			for field_name, fieldcontent in fieldcontents_in_one_set.items():
				data[set_name][field_name] = fieldcontent.get_data()
		return data

	def _build_vocabs(self):
		'''Invoke build vocab for each vocabulary'''
		for vocab in self.vocabs:
			vocab.build_vocab()

	def _collect_vocabs_from_fields(self, fields: Dict[str, OrderedDictType[str, Field]])\
			-> List[Vocab]:
		'''Collect all vocabulary instances (deduplicated).

		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, Field]]): field for each set.
		'''
		vocabs: List[Vocab] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, fields in fields_in_one_set.items():
				vocab = fields.get_vocab()
				if vocab is not None and vocab not in vocabs:
					vocabs.append(vocab)
		return vocabs

	def _collect_tokenizers_from_fields(self, fields: Dict[str, OrderedDictType[str, Field]])\
			-> List[Tokenizer]:
		'''Collect all tokenizer instances (deduplicated).

		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, Field]]): field for each set.
		'''
		tokenizers: List[Tokenizer] = []
		tokenizers_setting_hash: List[str] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, field in fields_in_one_set.items():
				tokenizer = field.get_tokenizer()
				if tokenizer is not None and tokenizer.get_setting_hash() not in tokenizers_setting_hash:
					tokenizers.append(tokenizer)
					tokenizers_setting_hash.append(tokenizer.get_setting_hash())
		return tokenizers

	def _fill_field_and_create_content(self, set_name: str, fields: \
				OrderedDictType[str, Union[str, Field]], \
				) -> \
					Tuple[OrderedDictType[str, Field], OrderedDictType[str, _FieldContent]]:
		'''Create and return fields and field contexts.

		Arguments:
			set_name(str): name of the set
			field (OrderedDictType[str, Union[str, Field]]): fields for the set.
		'''

		fieldcontents: OrderedDictType[str, _FieldContent] = OrderedDict()
		new_fields: OrderedDictType[str, Field] = OrderedDict()

		for name, field_name in fields:
			if isinstance(field_name, str):
				field = Field.load_class(field_name)()
			elif isinstance(field_name, Field):
				field = field_name
			fieldcontent = field._create(set_name) #pylint: disable=protected-access
			fieldcontents[name] = fieldcontent
			new_fields[name] = field
		return new_fields, fieldcontents

	def _create_data_hash(self, fieldcontents):
		raw_data_hash = sha256()
		data_hash = sha256()
		for _, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			for _, fieldcontent in fieldcontents_in_one_set.items():
				raw_data_hash.update(dumps(fieldcontent.get_raw_data_hash()))
				data_hash.update(dumps(fieldcontent.get_data_hash()))
		return raw_data_hash.hexdigest(), data_hash.hexdigest()

	def _create_setting_hash(self):
		setting_hash = sha256()
		for _, fields_in_one_set in self.fields.items():
			for _, field in fields_in_one_set.items():
				setting_hash.update(dumps(field._get_setting_hash(self.tokenizers, self.vocabs))) #pylint: disable=protected-access
		for vocab in self.vocabs:
			setting_hash.update(dumps(vocab.get_setting_hash()))
		for tokenizer in self.tokenizers:
			setting_hash.update(dumps(tokenizer.get_setting_hash()))
		return setting_hash.hexdigest()

	def _create_vocab_hash(self):
		vocab_hash = sha256()
		for vocab in self.vocabs:
			vocab_hash.update(dumps(vocab.get_vocab_hash()))
		return vocab_hash.hexdigest()

	def get_default_vocab(self) -> Vocab:
		'''Get the default :class:`Vocab` in this dataloader.
		TODO:
		# If there is only one vocabulary in the dataloader,
		# return it; otherwise, it can be set by :meth:`set_default_vocab` or
		# :meth:`.set_default_field`.
		'''
		vocab = self.get_default_field().get_vocab()
		if vocab is None:
			raise ValueError("This field do not have vocab")
		return vocab

	# def set_default_vocab(self, obj: Vocab):
	# 	'''Set the default vocabulary for this dataloader.
	# 	TODO: find the related function.

	# 	Arguments:
	# 		obj (:class:`Vocab`): the default vocabulary instance.
	# 	'''
	# 	try:
	# 		self.default_vocab_id = self.vocabs.index(obj)
	# 	except ValueError:
	# 		raise ValueError("obj must be one of the vocabulary instance in this dataloader.")

	def get_default_tokenizer(self) -> Tokenizer:
		'''Get the default :class:`Tokenizer` in this dataloader.
		TODO:
		# If there is only one tokenizer in the dataloader,
		# return it; otherwise, it can be set by :meth:`set_default_tokenizer` or
		# :meth:`.set_default_field`.
		'''
		tokenizer = self.get_default_field().get_tokenizer()
		if tokenizer is None:
			raise ValueError("This field do not have tokenizer")
		return tokenizer
		# if self.default_tokenizer_id is None:
		# 	raise RuntimeError("The dataloader has multiple tokenizers. \
		# 		Specify the default tokenizers by set_default_tokenizer.")
		# return self.tokenizers[self.default_tokenizer_id]

	# def set_default_tokenizer(self, obj: BaseTokenizer):
	# 	'''Set the default tokenizer instance for this dataloader.
	# 	TODO: find the related function.

	# 	Arguments:
	# 		obj (:class:`BaseTokenizer`): the default vocabulary instance.
	# 	'''
	# 	self.default_tokenizer_id = self.tokenizers.index(obj)

	def get_default_field(self) -> Field:
		'''Get the default :class:`Field` in this dataloader.
		It can be set by :meth:`.set_default_field`.
		'''
		if self.default_field_name is None or self.default_field_set_name is None:
			raise RuntimeError("No default field. \
				Specify the default field by set_default_field.")
		return self.fields[self.default_field_set_name][self.default_field_name]

	SET_NAME_DESCRIPTION = '''set_name (str): The name of set. For example: "train", "dev", "test".'''
	FIELD_NAME_DESCRIPTION = '''field_name (str): The name of field.'''

	def set_default_field(self, set_name: str, field_name: str):
		'''Set the default :class:`Field` in this dataloader. In the meanwhile,
		the default :class:`Vocab` and :class:`BaseTokenizer` is also set according
		to the field (if the field have vocab and tokenizer).
		TODO: find the related function.

		Arguments:
			{SET_NAME_DESCRIPTION}
			{FIELD_NAME_DESCRIPTION}
		'''
		if set_name not in self.fields:
			raise KeyError("No such set named %s" % set_name)
		elif field_name not in self.fields[set_name]:
			raise KeyError("No such field named %s" % field_name)
		self.default_field_set_name = set_name
		self.default_field_name = field_name

		# tokenizer = self.fields[set_name][field_name].get_tokenizer()
		# if tokenizer:
		# 	self.set_default_tokenizer(tokenizer)
		# vocab = self.fields[set_name][field_name].get_vocab()
		# if vocab:
		# 	self.set_default_vocab(vocab)

	def get_field(self, set_name: str, field_name: str) -> Field:
		'''Get :class:`Field` according to name of set and field.

		Arguments:
			{SET_NAME_DESCRIPTION}
			{FIELD_NAME_DESCRIPTION}
		'''
		return self.fields[set_name][field_name]

	def get_general_hash(self) -> str:
		'''General hash, identifying all details
		including raw data before processed, tokenized data, vocabulary, and settings.
		#TODO: write explaination of hash
		'''
		general_hash = sha256()
		general_hash.update(dumps(self._raw_data_hash))
		general_hash.update(dumps(self._data_hash))
		general_hash.update(dumps(self._vocab_hash))
		general_hash.update(dumps(self._setting_hash))
		return general_hash.hexdigest()

	def get_raw_data_hash(self) -> str:
		'''Raw data hash, identifying raw data before processed.
		#TODO: see explaination of hash
		'''
		return self._raw_data_hash

	def get_data_hash(self) -> str:
		'''Data hash, identifying data after processed (tokenized).
		#TODO: see explaination of hash
		'''
		return self._data_hash

	def get_vocab_hash(self) -> str:
		'''Vocab hash, identifying vocabulary.
		#TODO: see explaination of hash
		'''
		return self._vocab_hash

	def get_setting_hash(self) -> str:
		'''Setting hash, identifying settings to create the data loader.
		#TODO: see explaination of hash
		'''
		return self._setting_hash

	def restart(self, set_name, batch_size=None, shuffle=True):
		'''Initialize batches. This function be called before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
			{SET_NAME_DESCRIPTION}
			batch_size (int): the number of sample in a batch.
				default: if ``None``, last ``batch_size`` is used.
			shuffle (bool): whether to shuffle the data. Default: ``True``.
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		if batch_size is None and self.batch_size[set_name] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			# rng_state = random.getstate()
			random.shuffle(self.index[set_name])
			# random.setstate(rng_state)

		self.batch_id[set_name] = 0
		if batch_size is not None:
			self.batch_size[set_name] = batch_size
		batch_size_div = self.batch_size[set_name]
		assert batch_size_div is not None
		print("%s set restart, %d batches and %d left" % (set_name, \
						len(self.index[set_name]) // batch_size_div, \
						len(self.index[set_name]) % batch_size_div))

	_GET_BATCH_MORE_DOC = "Return a dict containing all the data from each field."
	_GET_BATCH_EXAMPLE = ""
	def get_batch(self, set_name: str, indexes: List[int]) -> Dict[str, Any]:
		'''Get a batch of data with specified `indexes`.
		{_GET_BATCH_MORE_DOC}

		Arguments:
			{SET_NAME_DESCRIPTION}
			indexes (list): a list of specified indexes of batched data.
		{_GET_BATCH_EXAMPLE}
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		res: Dict[str, Any] = {}
		for field_name, field_obj in self.fields[set_name]:
			res.update(field_obj._get_batch(field_name, self.data[set_name][field_name], indexes)) #pylint: disable=protected-access
		return res

	def get_next_batch(self, set_name, ignore_left_samples=False) -> Optional[Dict[str, Any]]:
		'''Get next batch. It can be called only after Initializing batches (:func:`restart`).
		Return a dict like :func:`get_batch`, or None if the epoch is end.

		Arguments:
			{SET_NAME_DESCRIPTION}
			ignore_left_samples (bool): If the number of left samples is not equal to
				``batch_size``, ignore them. This make sure all batches have same number of samples.
				Default: ``False``
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		batch_size = self.batch_size[set_name]
		if batch_size is None:
			raise RuntimeError( \
				"Please run restart before calling this function.")
		batch_id = self.batch_id[set_name]

		start, end = batch_id * \
					 	batch_size, (batch_id + 1) * batch_size
		if start >= len(self.index[set_name]):
			return None
		if ignore_left_samples and end > len(self.index[set_name]):
			return None
		index = self.index[set_name][start:end]
		res = self.get_batch(set_name, index)
		self.batch_id[set_name] += 1
		return res

	def get_batches(self, set_name, batch_size=None, shuffle=True,
			ignore_left_samples=False) -> Iterable[Dict[str, Any]]:
		'''An iterator over batches. It first call :func:`restart`, and then :func:`get_next_batches`
			until no more data is available. Returns an iterator where each element is like :func:`get_batch`.

		Arguments:
			{SET_NAME_DESCRIPTION}
			batch_size (int, optional): default: ``None``.  Use ``batch_size`` by default.
			shuffle (bool): whether to shuffle the data. Default: ``True``.
			ignore_left_samples (bool): If the number of left samples is not equal to
				``batch_size``, ignore them. This make sure all batches have same number of samples.
				Default: ``False``.
		'''
		self.restart(set_name, batch_size, shuffle)
		while True:
			res = self.get_next_batch(set_name, ignore_left_samples)
			if res is None:
				break
			yield res

	def get_all_batch(self, set_name):
		r'''Concatenate all batches to a single dict, where padding will not be applied.
		Returns a dict like :func:`get_batch`, but all the values are not padded
		and their type will be converted to list.

		Exactly, this function called :func:`.get_batch` where ``len(indexes)==1`` multiple times
		and concatenate all the values in the returned dicts.

		Arguments:
			{SET_NAME_DESCRIPTION}
		'''
		res: Dict[str, List[Any]] = {}
		for idx in self.index[set_name]:
			batch = self.get_batch(set_name, [idx])
			for attr, val in batch.items():
				if attr not in res:
					res[attr] = []
				if not isinstance(val, (list, np.ndarray)):
					val = [val]
				res[attr].extend(val)
		return res

	# copy some functions from vocab
	_VOCAB_MORE_DOCSTRING = '''It calls the method with the identical name of the :class:`Vocab` instance,\
		from ``self.get_default_vocab()``.'''
	frequent_vocab_size = copy_property(get_default_vocab, Vocab, "frequent_vocab_size")
	all_vocab_size = copy_property(get_default_vocab, Vocab, "all_vocab_size")
	frequent_vocab_list = copy_property(get_default_vocab, Vocab, "frequent_vocab_list")
	all_vocab_list = copy_property(get_default_vocab, Vocab, "all_vocab_list")
	get_special_tokens_mapping = copy_func(get_default_vocab, Vocab, "get_special_tokens_mapping")
	get_special_tokens_id = copy_func(get_default_vocab, Vocab, "get_special_tokens_id")
	pad_id = copy_property(get_default_vocab, Vocab, "pad_id")
	unk_id = copy_property(get_default_vocab, Vocab, "unk_id")
	go_id = copy_property(get_default_vocab, Vocab, "go_id")
	eos_id = copy_property(get_default_vocab, Vocab, "eos_id")

	_SENTENCE_MORE_DOCSTRING = '''It calls the method with the identical name of the :class:`Sentence` instance,\
		from ``self.get_default_field()``.'''
	tokenize = copy_func(get_default_field, Sentence, "tokenize")
	tokenize_sentences = copy_func(get_default_field, Sentence, "tokenize_sentences")
	convert_tokens_to_ids = copy_func(get_default_field, Sentence, "convert_tokens_to_ids")
	convert_ids_to_tokens = copy_func(get_default_field, Sentence, "convert_ids_to_tokens")
	convert_ids_to_sentence = copy_func(get_default_field, Sentence, "convert_ids_to_sentence")
	convert_sentence_to_ids = copy_func(get_default_field, Sentence, "convert_sentence_to_ids")
	add_special_to_ids = copy_func(get_default_field, Sentence, "add_special_to_ids")
	remove_special_in_ids = copy_func(get_default_field, Sentence, "remove_special_in_ids")
	process_sentences = copy_func(get_default_field, Sentence, "process_sentences")
	trim_in_ids = copy_func(get_default_field, Sentence, "trim_in_ids")
