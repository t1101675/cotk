'''A module for vocab'''
from typing import Optional, List, Dict, Any
from collections import Counter, OrderedDict
from itertools import chain
import logging
import hashlib

from .._utils.typehint import OrderedDictType
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from .._utils.unordered_hash import dumps
from .context import VocabContext
from .tokenizer import PretrainedTokenizer


class Vocab(LoadClassInterface, metaclass=DocStringInheritor):
	'''A class for storing vocabulary.
	TODO: see explanation'''
	def __init__(self):
		self._setting_hash: Optional[str] = None

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		'''Add tokens for this vocabulary instance, the tokens will be used for building
		vocabulary list. Must be called before :meth:`.build_vocab`.
		See TODO: explain the vocab_from.

		Arguments:
			tokens (List[str]): A list of tokens to add to the vocabulary.
			vocab_from (str): One of ``train``, ``test``, ``extra`` or ``default``. \
				If ``train``, the tokens are from the training data. \
				If ``test``, the tokens are from the validation data or test data. \
				If ``ignore`` or ``default``, the tokens are from extra data.
		'''
		raise NotImplementedError

	def build_vocab(self):
		'''Building the vocabulary list according to the tokens from :meth:`.add_tokens`.
		See TODO: explain how to build vocab.
		'''
		raise NotImplementedError

	_VOCAB_MORE_DOCSTRING = ""
	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		'''Convert list of tokens to list of ids. {_VOCAB_MORE_DOCSTRING}

		Arguments:
			tokens (List[str]): List of tokens.
			only_frequent_word (bool, optional): Use ``unk`` for rare tokens. Defaults: False.

		Returns:
			List[int]: The corresponding list of ids.
		'''
		raise NotImplementedError

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		'''Convert list of ids to list of tokens. {_VOCAB_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): List of ids.

		Returns:
			List[str]: The Corresponding list of tokens.
		'''
		raise NotImplementedError

	@property
	def frequent_vocab_size(self):
		'''int: The number of **frequent** words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def all_vocab_size(self):
		'''int: The number of frequent words and rare words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def frequent_vocab_list(self):
		'''list: The list of frequent words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def all_vocab_list(self):
		'''list: The list of frequent words and rare words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	SPECIAL_TOKEN_DOCS = '''Special tokens mapping is an ordered dict \
		mapping the general name of special tokens to its string. \
		The key must be one of the following: \
		``pad``, ``unk``, ``go``, ``eos``, ``sep``, ``cls``, ``mask``. \
		The value can be arbitrary string, e.g., ``"<pad>"``, ``"<unk>"``.'''

	def get_special_tokens_mapping(self) -> OrderedDictType[str, str]:
		'''Get special tokens mapping. {SPECIAL_TOKEN_DOCS} {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	def get_special_tokens_id(self, name: str) -> int:
		'''Get id of special token specifying the general name.
		Raise KeyError if no such token in this instance. {_VOCAB_MORE_DOCSTRING}

		Arguments:
			name (str): the general name, must be one of the following,
				``pad``, ``unk``, ``go``, ``eos``, ``sep``, ``cls``, ``mask``.
		'''
		raise NotImplementedError

	@property
	def pad_id(self) -> int:
		'''int: The id of pad token. Raise KeyError if no pad token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		return self.get_special_tokens_id("pad")
	@property
	def unk_id(self) -> int:
		'''int: The id of unk token. Raise KeyError if no unk token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		return self.get_special_tokens_id("unk")
	@property
	def go_id(self) -> int:
		'''int: The id of go token. Raise KeyError if no go token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		return self.get_special_tokens_id("go")
	@property
	def eos_id(self) -> int:
		'''int: The id of eos token. Raise KeyError if no eos token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		return self.get_special_tokens_id("eos")

	def get_setting_hash(self) -> str:
		'''Get setting hash for the Vocabulary instance.
		See :ref:`here <dataloader_hash>` for the explaination of ``setting hash``.

		Returns:
			str: the setting hash.
		'''
		assert self._setting_hash is not None
		return self._setting_hash

	def get_vocab_hash(self) -> str:
		'''Get vocab hash for the Vocabulary instance.
		See :ref:`here <dataloader_hash>` for the explaination of ``vocab hash``.

		Returns:
			str: the vocab hash.
		'''
		raise NotImplementedError

class GeneralVocab(Vocab):
	'''A vocabulary class for general use.	If any argument is not specified,
	the value will be first retrieved from :class:`VocabContext`. If still ``None``, default
	value will be used.

	Arguments:
		min_frequent_vocab_times (int, optional): Tokens from training data appeared \
			more than ``min_frequent_vocab_times`` will be regarded as frequent words. \
			If None, it will be ``0``.
		min_rare_vocab_times (int, optional): Tokens from training data or test data \
			appeared more than ``min_rare_vocab_times`` will be regarded as rare words \
			(frequent word excluded). If None, it will be ``0``.
		special_tokens_mapping (OrderedDict, optional): {SPECIAL_TOKEN_DOCS_WITH_DEFAULT}
		special_appeared_in_data (bool, optional): If the string of special tokens will \
			appear in the data. If None, it will be ``False``.
	'''

	SPECIAL_TOKEN_DOCS_WITH_DEFAULT = Vocab.SPECIAL_TOKEN_DOCS + ''' \
			If None, it will be ``OrderedDict([("pad", "<pad>"), ("unk", "<unk>"),\
			("go", "<go>"), ("eos", "<eos>")])``.'''

	def __init__(self, min_frequent_vocab_times: Optional[int] = None, \
			min_rare_vocab_times: Optional[int] = None, \
			special_tokens_mapping: Optional[OrderedDictType[str, str]] = None, \
			special_appeared_in_data: Optional[bool] = None):
		super().__init__()

		with VocabContext.set_parameters(\
				min_frequent_vocab_times=min_frequent_vocab_times,\
				min_rare_vocab_times=min_rare_vocab_times,\
				special_tokens_mapping=special_tokens_mapping,\
				special_appeared_in_data=special_appeared_in_data):
			self.min_frequent_vocab_times: int = VocabContext.get("min_frequent_vocab_times", 0)
			self.min_rare_vocab_times: int = VocabContext.get("min_rare_vocab_times", 0)
			filled_special_tokens: Optional[OrderedDictType[str, str]] = VocabContext.get("special_tokens_mapping", None)
			self.special_appeared_in_data: bool = VocabContext.get("special_appeared_in_data", False)

		self.special_tokens_mapping = filled_special_tokens or OrderedDict(
			[("pad", "<pad>"), ("unk", "<unk>"), ("go", "<go>"), ("eos", "<eos>")]
		)

		self.mode = "init"
		self.train_tokens: Optional[List[str]] = []
		self.test_tokens: Optional[List[str]] = []

		self._all_vocab_list: Optional[List[str]] = None
		self.word2id: Optional[Dict[str, int]] = None
		self._frequent_vocab_size: int = 0

		self._setting_hash = hashlib.sha256(dumps([ \
			"Vocab", \
			"configs", \
			self.min_frequent_vocab_times, \
			self.min_rare_vocab_times, \
			self.special_tokens_mapping, \
			self.special_appeared_in_data \
		])).hexdigest()

	@staticmethod
	def from_predefined(vocab_list: List[str], \
		frequent_vocab_size: int, \
		special_tokens_mapping: Optional[OrderedDictType[str, str]] = None) -> "GeneralVocab":
		'''Initialize the vocabulary from a predefined list. See :meth:`.from_predefined_vocab` if
		you want to use the vocabulary from an existing :class:`GeneralVocab` instance.

		Arguments:
			vocab_list (List[str]): A list of all vocabulary.
			frequent_vocab_size (int): the length of the frequent words. \
				The frequent word must be in the front of the ``vocab_list``.
			special_tokens_mapping (OrderedDict, optional): {SPECIAL_TOKEN_DOCS_WITH_DEFAULT} \
				Notice special tokens should be in the front of the ``vocab_list`` (included in frequent words, \
				ordered sensitive).
		Returns:
			:class:`GeneralVocab`: A GeneralVocab instance with specified vocabulary.
		'''
		vocab = GeneralVocab(special_tokens_mapping=special_tokens_mapping)
		special_values = list(vocab.get_special_tokens_mapping().values())
		if vocab_list[:len(special_values)] != special_values:
			raise ValueError("special tokens should be in the front of the vocab_list, where special tokens are %s, but \
				the first tokens of vocab_list are %s." %
				(repr(special_values), repr(vocab_list[:len(special_values)]))
			)
		if len(set(vocab_list)) != len(vocab_list):
			raise ValueError("vocab_list should not contain a single token multiple times")

		#pylint: disable=protected-access
		vocab.mode = "finish"
		vocab._all_vocab_list = vocab_list
		vocab._frequent_vocab_size = frequent_vocab_size
		vocab.word2id = {w: i for i, w in enumerate(vocab.all_vocab_list)}

		vocab.train_tokens = None
		vocab.test_tokens = None

		vocab._setting_hash = hashlib.sha256(dumps([ \
			"Vocab", \
			"predefined", \
			vocab.all_vocab_list, \
			vocab._frequent_vocab_size, \
			len(vocab.special_tokens_mapping) \
		])).hexdigest()

		return vocab

	@staticmethod
	def from_predefined_vocab(vocab: "GeneralVocab") -> "GeneralVocab":
		'''Init a new :class:`GeneralVocab` instance from ``vocab``. The new instance have the same
		vocabulary list as the old one.

		Arguments:
			vocab(:class:`GeneralVocab`): The old instance.

		Returns:
			:class:`GeneralVocab`: The new instance that have the same vocabulary with the old one.
		'''
		if not isinstance(vocab, GeneralVocab):
			raise TypeError("vocab must be an instance of GeneralVocab class.")
		vocab_list = vocab.all_vocab_list
		frequent_vocab_size = vocab._frequent_vocab_size
		special_token_mappings = vocab.get_special_tokens_mapping()
		return GeneralVocab.from_predefined(vocab_list, frequent_vocab_size, special_token_mappings)


	@staticmethod
	def from_frequent_word(frequent_vocab_list: List[str], \
			special_tokens_mapping: Optional[OrderedDictType[str, str]] = None) -> "GeneralVocab":
		'''Initialize the vocabulary from a predefined frequent list, and the rare word list can be built later.
		See :meth:`.from_frequent_word_of_vocab`
		if you want to use the frequent vocabulary from an existing :class:`GeneralVocab` instance.

		Arguments:
			frequent_vocab_list (List[str]): A list of frequent vocabulary.
			special_tokens_mapping (OrderedDict, optional): {SPECIAL_TOKEN_DOCS_WITH_DEFAULT} \
				Notice special tokens should be in the front of the ``frequent_vocab_list`` (ordered sensitive).
		Returns:
			:class:`GeneralVocab`: A GeneralVocab instance with specified vocabulary.
		'''

		vocab = GeneralVocab(special_tokens_mapping=special_tokens_mapping)
		special_values = list(vocab.get_special_tokens_mapping().values())
		if frequent_vocab_list[:len(special_values)] != special_values:
			raise ValueError("special tokens should be in the front of the vocab_list, where special tokens are %s, but \
				the first tokens of vocab_list are %s." %
				(repr(special_values), repr(frequent_vocab_list[:len(special_values)]))
			)

		#pylint: disable=protected-access
		vocab.mode = "frequent_specified"
		vocab._all_vocab_list = frequent_vocab_list

		vocab._setting_hash = hashlib.sha256(dumps([ \
			"Vocab", \
			"frequent", \
			frequent_vocab_list, \
			special_tokens_mapping \
		])).hexdigest()
		return vocab

	@staticmethod
	def from_frequent_word_of_vocab(vocab: "GeneralVocab") -> "GeneralVocab":
		'''Init a new :class:`Vocab` instance from the frequent words of ``vocab``. The new instance have the same
		frequent vocabulary list as the old one. The rare word list can be built later.

		Arguments:
			vocab(:class:`GeneralVocab`): The old instance to provide frequent words.

		Returns:
			:class:`GeneralVocab`: The new instance that have the same frequent vocabulary with the old one.
		'''
		if not isinstance(vocab, GeneralVocab):
			raise TypeError("vocab must be an instance of GeneralVocab class.")
		vocab_list = vocab.all_vocab_list
		frequent_vocab_size = vocab.frequent_vocab_size
		special_token_mappings = vocab.get_special_tokens_mapping()
		return GeneralVocab.from_predefined(vocab_list[:frequent_vocab_size], special_token_mappings)

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		if self.train_tokens is None or self.test_tokens is None:
			return
			#raise RuntimeError("Vocabulary has been built, cannot add more tokens.")
		if vocab_from == "train":
			self.train_tokens.extend(tokens)
		elif vocab_from == "test":
			self.test_tokens.extend(tokens)
		elif vocab_from in ["extra", "default"]:
			pass
		else:
			raise ValueError("Unknown vocab_from: %s, only supports frequent, rare, extra or default" % vocab_from)

	def build_vocab(self) -> None:
		if self.mode == "finish":
			return
			#raise RuntimeError("Vocabulary has been built, cannot build again.")
		if self.train_tokens is None or self.test_tokens is None:
			raise RuntimeError("Train tokens or test toktens should not be None")

		if not self.special_appeared_in_data:
			all_token_set = set(chain(self.train_tokens, self.test_tokens))
			for special_token in self.special_tokens_mapping.values():
				if special_token in all_token_set:
					raise RuntimeError("Dataset file contains special tokens %d. If it is desired, try to set \
						'special_appeared_in_data' to True in Vocab or Dataloader.")

		exclude_set = set(self.special_tokens_mapping.values())
		if self.mode != "frequent_specified":
			assert self.all_vocab_list is None
			vocab = sorted(Counter(self.train_tokens).most_common(), \
						key=lambda pair: (-pair[1], pair[0]))
			frequent_vocab = [x[0] for x in vocab if x[1] >= self.min_frequent_vocab_times and x[0] not in exclude_set]
		else:
			assert self.all_vocab_list is not None
			frequent_vocab = self.all_vocab_list

		exclude_set.update(frequent_vocab)
		vocab = sorted(Counter(chain(self.train_tokens, self.test_tokens)).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		rare_vocab = [x[0] for x in vocab if x[1] >= self.min_rare_vocab_times \
				and x[0] not in exclude_set]

		self._all_vocab_list = list(self.special_tokens_mapping.values()) + frequent_vocab + rare_vocab
		self._frequent_vocab_size = len(self.special_tokens_mapping) + len(frequent_vocab)

		logging.info("frequent vocab list length = %d", self._frequent_vocab_size)
		logging.info("frequent + rare vocab list length = %d", len(self._all_vocab_list))

		self.word2id = {w: i for i, w in enumerate(self._all_vocab_list)}

		self.train_tokens = None
		self.test_tokens = None
		self.mode = "finish"

	_VOCAB_MORE_DOCSTRING = ""
	def get_special_tokens_id(self, name) -> int:
		try:
			return self.word2id[self.special_tokens_mapping[name]] # type: ignore
		except KeyError:
			raise KeyError("No such special token in this class")

	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		if self.word2id is None:
			raise RuntimeError("You have to run build_vocab first")
		ids = [self.word2id.get(token, self.unk_id) for token in tokens]
		if only_frequent_word:
			ids = [self.unk_id if i >= self._frequent_vocab_size else i for i in ids]
		return ids

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		if self._all_vocab_list is None:
			raise RuntimeError("You have to run build_vocab first")
		return [self._all_vocab_list[word] for word in ids]

	def get_vocab_hash(self) -> str:
		return hashlib.sha256(dumps([ \
				self._all_vocab_list, \
				self._frequent_vocab_size, \
				len(self.special_tokens_mapping) \
			])).hexdigest()

	@property
	def frequent_vocab_size(self):
		return self._frequent_vocab_size

	@property
	def all_vocab_size(self):
		return len(self._all_vocab_list) # type: ignore

	@property
	def frequent_vocab_list(self):
		'''list: The list of frequent words. Special tokens are always in the front of the list.
		'''
		return self._all_vocab_list[:self._frequent_vocab_size] # type: ignore

	@property
	def all_vocab_list(self):
		'''list: The list of frequent and rare words. Frequent words are always in the front of the list.
		'''
		return self._all_vocab_list[:] # type: ignore

	def get_special_tokens_mapping(self):
		return self.special_tokens_mapping

class PretrainedVocab(Vocab):
	'''Use the vocabulary from a pretrained tokenizer in transformers package.
	This class is usually used for pretrained models, and it do not have rare words.

	Arguments:
		tokenizer (``transformers.Pretrainedtokenizer``): A pretrained tokenizer from transformers package.
	'''
	def __init__(self, tokenizer: Any):
		super().__init__()
		self.tokenizer = PretrainedTokenizer(tokenizer)
		self._inner_tokenizer = tokenizer
		self._setting_hash = hashlib.sha256(dumps(["pretrained", self.tokenizer.get_setting_hash()])).hexdigest()

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		pass

	def build_vocab(self) -> None:
		pass

	def get_vocab_hash(self):
		return self._setting_hash #vocab hash is represented by tokenizer

	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		return self._inner_tokenizer.convert_tokens_to_ids(tokens)

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		return self._inner_tokenizer.convert_ids_to_tokens(ids)

	@property
	def frequent_vocab_size(self):
		return self._inner_tokenizer.vocab_size

	@property
	def all_vocab_size(self):
		return self._inner_tokenizer.vocab_size

	@property
	def frequent_vocab_list(self):
		'''list: The list of vocabulary list. This class do not have rare words, thus the return value
		is always the same with :meth:`all_vocab_list`.
		'''
		return self.convert_ids_to_tokens(list(range(self.frequent_vocab_size)))

	@property
	def all_vocab_list(self):
		'''list: The list of vocabulary list. This class do not have rare words, thus the return value
		is always the same with :meth:`frequent_vocab_list`.
		'''
		return self.frequent_vocab_list

	def get_special_tokens_mapping(self):
		old_key = ["pad_token", "unk_token", "bos_token", "eos_token", "sep_token", "cls_token", "mask_token"]
		new_key = ["pad", "unk", "go", "eos", "sep", "cls", "mask"]
		res = OrderedDict()
		for key, value in self._inner_tokenizer.special_tokens_map.items():
			if key in old_key:
				idx = old_key.index(key)
				res[new_key[idx]] = value
		return res

	_VOCAB_MORE_DOCSTRING = ""
	def get_special_tokens_id(self, name):
		try:
			return self.convert_tokens_to_ids([self.get_special_tokens_mapping()[name]])[0]
		except KeyError:
			raise KeyError("No such special token in this class")

#TODO: add a class for sparse label. It convert sparse tokens to id, but it do not use special tokens,
# frequent tokens or more.
class SimpleVocab(Vocab):
	pass
