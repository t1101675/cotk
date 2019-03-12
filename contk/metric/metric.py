r"""
``contk.metrics`` provides classes and functions evaluating results of models. It provides
a fair metric for every model.
"""
import random
import multiprocessing
from multiprocessing import Pool
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from .._utils.unordered_hash import UnorderedSha256

class MetricBase:
	'''Base class for metrics.
	'''
	def __init__(self):
		pass

class _PrecisionRecallMetric(MetricBase):
	'''Base class for precision recall metrics. This is an abstract class.
	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_allvocabs_key (str): Reference sentences are passed to :func:`forward` by
			``data[reference_allvocabs_key]``. Default: ``resp_allvocabs``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.

	Attributes:
		res_prefix (str): Prefix added to the front of each key
						in the result dict of ^close^
	'''
	def __init__(self, dataloader, reference_allvocabs_key='resp_allvocabs', gen_key='gen'):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.prec_list = []
		self.rec_list = []
		self.res_prefix = ""

	def score(self, gen, reference):
		r'''This function is called by ^forward^

		Arguments:
			* gen (list): list of generated word ids
			* reference (list): list of word ids of a reference

		Returns:
			(scalar): score \in [0, 1]
		'''
		raise NotImplementedError( \
			"This function should be implemented by subclasses.")

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list of list of list): Reference sentences.
				Does not contain start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Outermost list: batch_size
				Innermost list: number of words, allow different sizes
				Second innermost list: number of sentences, allow different sizes
			data[gen_prob_key] (list of list of list): Sentence generations model outputs
				similar to data[reference_allvocabs_key]
		'''
		references = data[self.reference_allvocabs_key]
		gens = data[self.gen_key]

		if len(references) != len(gens):
			raise ValueError("Batch num is not matched.")

		for reference, gen in zip(references, gens):
			# pylint: disable=no-member
			matrix = np.zeros((len(reference), len(gen)), dtype=np.float32)
			for i, single_ref in enumerate(reference):
				for j, single_gen in enumerate(gen):
					matrix[i][j] = self.score(single_gen, single_ref)
			self.prec_list.append(float(np.sum(np.max(matrix, 0))) / len(gen))
			self.rec_list.append(float(np.sum(np.max(matrix, 1))) / len(references))

	def close(self):
		'''Return a dict which contains:

			* **precision**: average precision
			* **recall**: average recall
		'''
		return {'{} precision'.format(self.res_prefix): np.average(self.prec_list), \
				'{} recall'.format(self.res_prefix): np.average(self.rec_list)}

class BleuPrecisionRecallMetric(_PrecisionRecallMetric):
	'''Metric for calculating sentence BLEU precision and recall

	Arguments:
		* ngram (int): Specifies BLEU-ngram
	'''
	def __init__(self, dataloader, ngram, reference_allvocabs_key='resp_allvocabs', gen_key='gen'):
		super().__init__(dataloader, reference_allvocabs_key, gen_key)
		if ngram not in range(1, 5):
			raise ValueError("ngram should belong to [1, 4]")
		self.ngram = ngram
		self.weights = [1 / ngram] * ngram
		self.res_prefix = 'BLEU-{}'.format(ngram)

	def score(self, gen, reference):
		r'''Score_fn of BLEU-ngram precision and recall

		Returns:
			(scalar): sentence bleu score \in [0, 1]
		'''
		return sentence_bleu([reference], gen, self.weights, SmoothingFunction().method1)

class EmbSimilarityPrecisionRecallMetric(_PrecisionRecallMetric):
	'''Metric for calculating cosine similarity precision and recall

	Arguments:
		* embed (:class:^numpy.array^): A 2-d padded array of word embeddings
		* mode (str): Specifies the operation that computes the bag-of-word representation.
					Must be 'avg' or 'extrema':
						'avg': element-wise average word embeddings
						'extrema': element-wise maximum word embeddings
	'''
	def __init__(self, dataloader, embed, mode, \
				reference_allvocabs_key='resp_allvocabs', gen_key='gen'):
		super().__init__(dataloader, reference_allvocabs_key, gen_key)
		if not isinstance(embed, np.ndarray) or len(np.shape(embed)) != 2:
			raise ValueError("invalid type or shape or embed.")
		if mode not in ['avg', 'extrema']:
			raise ValueError("mode should be 'avg' or 'extrema'.")
		if len(embed) != self.dataloader.vocab_size:
			raise ValueError("embed size not equal to vocab size.")
		self.embed = embed
		self.mode = mode
		self.res_prefix = '{}-bow'.format(mode)

	def score(self, gen, reference):
		r'''Score_fn of cosine similarity precision and recall

		Returns:
			(Scalar): cosine similarity between two sentence embeddings \in [0, 1]
		'''
		gen_vec = []
		ref_vec = []
		for i in gen:
			if i < 0:
				raise ValueError("gen index out of range.")
			elif i >= self.dataloader.vocab_size:
				gen_vec.append(self.embed[self.dataloader.unk_id])
			else:
				gen_vec.append(self.embed[i])
		for i in reference:
			if i < 0:
				raise ValueError("reference index out of range.")
			elif i >= self.dataloader.vocab_size:
				ref_vec.append(self.embed[self.dataloader.unk_id])
			else:
				ref_vec.append(self.embed[i])
		if self.mode == 'avg':
			gen_embed = np.average(gen_vec, 0)
			ref_embed = np.average(ref_vec, 0)
		else:
			gen_embed = np.max(gen_vec, 0)
			ref_embed = np.max(ref_vec, 0)
		cos = np.sum(gen_embed * ref_embed) / \
			  np.sqrt(np.sum(gen_embed * gen_embed) * np.sum(ref_embed * ref_embed))
		norm = (cos + 1) / 2
		return norm

class PerplexityMetric(MetricBase):
	'''Metric for calculating perplexity.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``resp_allvocabs``.
		reference_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[reference_len_key]``. Default: ``resp_length``.
		gen_log_prob_key (str): Sentence generations model outputs of **log softmax** probability
			are passed to :func:`forward` by ``data[gen_log_prob_key]``. Default: ``gen_log_prob``.
		invalid_vocab (bool): whether gen_log_prob contains invalid vocab. Default: False
		full_check (bool): whether perform full checks on `gen_log_prob` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			Default: False
	'''
	def __init__(self, dataloader, \
					   reference_allvocabs_key="resp_allvocabs", \
					   reference_len_key="resp_length", \
					   gen_log_prob_key="gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.word_loss = 0
		self.length_sum = 0
		self.invalid_vocab = invalid_vocab
		self.full_check = full_check

	def forward(self, data):
		'''Processing a batch of data. Smoothing will be performed for invalid vocabs.
		Unknowns vocabs will be ignored.

		TODO:
			Find a place to explain valid vocabs, invalid vocabs, and unknown vocabs.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`): Reference sentences with all vocabs
				with all vocabs. Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[reference_len_key] (list): Length of Reference sentences. Contains start token (eg:``<go>``)
				and end token (eg:``<eos>``). Size: `[batch_size]`
			data[gen_log_prob_key] (list or :class:`numpy.array`): Sentence generations model outputs of
				**log softmax** probability. Contains end token (eg:``<eos>``), but without start token
				(eg: ``<go>``).	The 2nd dimension can be jagged.
				Size: `[batch_size, gen_sentence_length, vocab_size]` for ``invalid_vocab = False``.
				`[batch_size, gen_sentence_length, all_vocab_size]` for ``invalid_vocab = True``.

		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		resp_allvocabs = data[self.reference_allvocabs_key]
		resp_length = data[self.reference_len_key]
		gen_log_prob = data[self.gen_log_prob_key]
		if len(resp_allvocabs) != len(resp_length) or len(resp_allvocabs) != len(gen_log_prob):
			raise ValueError("Batch num is not matched.")

		# perform random check to assert the probability is valid
		checkid = random.randint(0, len(resp_length)-1)
		if resp_length[checkid] < 2:
			raise ValueError("resp_length must no less than 2, because <go> and <eos> are always included.")
		checkrow = random.randint(0, resp_length[checkid]-2)
		if not np.isclose(np.sum(np.exp(gen_log_prob[checkid][checkrow])), 1):
			print("gen_log_prob[%d][%d] exp sum is equal to %f." % (checkid, checkrow, \
				np.sum(np.exp(gen_log_prob[checkid][checkrow]))))
			raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

		if not isinstance(resp_allvocabs, np.ndarray):
			resp_allvocabs = np.array(resp_allvocabs)
		if not isinstance(gen_log_prob, np.ndarray):
			gen_log_prob = np.array(gen_log_prob)

		invalid_vocab_num = self.dataloader.all_vocab_size - self.dataloader.vocab_size
		#resp = resp_allvocabs.copy()
		#resp[resp >= self.dataloader.vocab_size] = self.dataloader.unk_id

		for i, single_length in enumerate(resp_length):
			# perform full check to assert the probability is valid
			if self.full_check:
				expsum = np.sum(np.exp(gen_log_prob[i][:single_length-1]), -1)
				if not np.allclose(expsum, [1] * (single_length - 1)):
					raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

			resp_now = np.array(resp_allvocabs[i][1:single_length])
			gen_log_prob_now = np.array(gen_log_prob[i])

			if not self.invalid_vocab:
				if gen_log_prob_now.shape[1] != self.dataloader.vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to vocab_size when \
						invalid_vocab = False, \
						but %d != %d" % (gen_log_prob_now.shape[1], self.dataloader.vocab_size))
			else:
				if gen_log_prob_now.shape[1] != self.dataloader.all_vocab_size:
					raise ValueError("The third dimension gen_log_prob should be equals to all_vocab_size \
						when invalid_vocab = True, \
						but %d != %d" % (gen_log_prob_now.shape[1], self.dataloader.vocab_size))

			# calc normal vocab
			normal_idx = np.where(np.logical_and(resp_now != self.dataloader.unk_id, \
									resp_now < self.dataloader.vocab_size))
			self.word_loss += -np.sum(gen_log_prob_now[normal_idx, resp_now[normal_idx]])
			self.length_sum += np.array(normal_idx).shape[1]
			# calc invalid vocab
			invalid_idx = np.where(resp_now >= self.dataloader.vocab_size)
			invalid_log_prob = gen_log_prob_now[\
									invalid_idx, [self.dataloader.unk_id] * len(invalid_idx) \
								] - np.log(invalid_vocab_num)
			if self.invalid_vocab:
				extra_invalid_log_prob = gen_log_prob_now[invalid_idx, resp_now[invalid_idx]]
				self.word_loss += -np.sum(np.log( \
						np.exp(invalid_log_prob) + np.exp(extra_invalid_log_prob) \
					))
			else:
				self.word_loss += -np.sum(invalid_log_prob)
			self.length_sum += np.array(invalid_idx).shape[1]

	def close(self):
		'''Return a dict which contains:

			* **perplexity**: perplexity value
		'''
		return {"perplexity": np.exp(self.word_loss / self.length_sum)}

class MultiTurnPerplexityMetric(MetricBase):
	'''Metric for calculating multi-turn perplexity.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``sent_allvocabs``.
		reference_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[reference_len_key]``. Default: ``sent_length``.
		gen_log_prob_key (str): Sentence generations model outputs of **log softmax** probability
			are passed to :func:`forward` by ``data[gen_log_prob_key]``. Default: ``gen_log_prob``.
		invalid_vocab (bool): whether gen_log_prob contains invalid vocab. Default: False
		full_check (bool): whether perform full checks on `gen_log_prob` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			Default: False
	'''
	def __init__(self, dataloader, reference_allvocabs_key="sent_allvocabs", \
					   reference_len_key="sent_length", \
					   gen_log_prob_key="gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.invalid_vocab = invalid_vocab
		self.sub_metric = PerplexityMetric(dataloader, \
				reference_allvocabs_key="sent_allvocabs", \
				reference_len_key="sent_length", \
				gen_log_prob_key="gen_log_prob", \
				invalid_vocab=invalid_vocab, \
				full_check=full_check)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`): Reference sentences
				with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[reference_len_key] (list of list): Length of Reference sentences. Contains
				start token (eg:``<go>``) and end token (eg:``<eos>``). It must NOT be padded,
				which means the inner lists may have different length.
				Length of outer list: `batch_size`
			data[gen_log_prob_key] (list or :class:`numpy.array`): Sentence generations model outputs of
				**log softmax** probability. Contains end token (eg:``<eos>``), but without start token
				(eg: ``<go>``).	The 2nd / 3rd dimension can be jagged or padded.
				Size: `[batch_size, max_turn_length, gen_sentence_length, vocab_size]`.

		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		reference_allvocabs = data[self.reference_allvocabs_key]
		length = data[self.reference_len_key]
		gen_log_prob = data[self.gen_log_prob_key]
		if len(length) != len(reference_allvocabs) or len(length) != len(gen_log_prob):
			raise ValueError("Batch num is not matched.")

		for i, sent_length in enumerate(length):
			# Pass turn as batch for sub_metric, the result will be same.
			turn_length = len(sent_length)
			if len(reference_allvocabs[i]) < turn_length or len(gen_log_prob[i]) < turn_length:
				raise ValueError("Turn num is not matched.")
			self.sub_metric.forward({"sent_allvocabs": reference_allvocabs[i][:turn_length], \
					"sent_length": sent_length, \
					"gen_log_prob": gen_log_prob[i][:turn_length]})

	def close(self):
		'''Return a dict which contains:

			* **perplexity**: perplexity value
		'''
		return self.sub_metric.close()

class BleuCorpusMetric(MetricBase):
	'''Metric for calculating BLEU.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:.forward by ``data[reference_allvocabs_key]``.
			Default: ``resp``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, reference_allvocabs_key="resp_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array` of `int`):
				reference_allvocabs sentences.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		resp = data[self.reference_allvocabs_key]
		if len(resp) != len(gen):
			raise ValueError("Batch num is not matched.")

		for gen_sen, resp_sen in zip(gen, resp):
			self.hyps.append(self.dataloader.trim_index(gen_sen))
			self.refs.append([self.dataloader.trim_index(resp_sen[1:])])

	def close(self):
		'''Return a dict which contains:

			* **bleu**: bleu value.
		'''
		try:
			return {"bleu": \
				corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method7)}
		except ZeroDivisionError as _:
			raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")


class SelfBleuCorpusMetric(MetricBase):
	'''Metric for calculating Self-BLEU.

	Arguments:
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
		sample (int): Number of samples sampled from the generated sentences. Default: 1000.
	'''
	def __init__(self, dataloader, gen_key="gen", sample=1000):
		super().__init__()
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.sample = sample
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]

		for gen_sen in gen:
			self.hyps.append(self.dataloader.trim_index(gen_sen))
	def run_f(self, ele):
		'''Auxiliary function which returns:
			* **sentence-self-bleu**: sentence-self-bleu value.
		'''
		return sentence_bleu(ele[0], ele[1], ele[2], smoothing_function=SmoothingFunction().method1)

	def close(self):
		'''Return a dict which contains:

			* **self-bleu**: self-bleu value.
		'''
		if self.sample > len(self.hyps):
			self.sample = len(self.hyps)
		random.shuffle(self.hyps)
		ref = self.hyps[:self.sample]

		try:
			result = {}
			for ngram in range(2, 5):
				weight = tuple((1. / ngram for _ in range(ngram)))
				if self.sample >= 1000:
					pool = Pool(multiprocessing.cpu_count())
					bleu_irl = pool.map(self.run_f, [(ref[:i]+ref[i+1:self.sample], ref[i], weight) \
										for i in range(self.sample)])
					pool.close()
					pool.join()
				else:
					bleu_irl = []
					for i in range(self.sample):
						bleu_irl.append(self.run_f((ref[:i]+ref[i+1:], ref[i], weight)))
				result["self-bleu-%d"%ngram] = 1.0 * sum(bleu_irl) / len(bleu_irl)
			return result
		except ZeroDivisionError as _:
			raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")

class FwBwBleuCorpusMetric(MetricBase):
	'''Metric for calculating BLEU.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_test_key (str): Reference sentences with all vocabs in test data
			are passed to :func:.forward by ``data[reference_test_key]``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
		sample (int): Number of samples sampled from the generated sentences. Default: 1000.
	'''
	def __init__(self, dataloader, \
			reference_test_key, \
			gen_key="gen", \
			sample=1000):
		super().__init__()
		self.dataloader = dataloader
		self.reference_test_key = reference_test_key
		self.gen_key = gen_key
		self.sample = sample

		self.refs = []
		self.hyps = []
		resp = self.dataloader.data["test"][self.reference_test_key]
		for resp_sen in resp:
			self.refs.append(self.dataloader.trim_index(resp_sen[1:]))

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]

		for gen_sen in gen:
			self.hyps.append(self.dataloader.trim_index(gen_sen))

	def run_f(self, ele):
		'''Auxiliary function which returns:
			* **sentence-self-bleu**: sentence-self-bleu value.
		'''
		return sentence_bleu(ele[0], ele[1], ele[2], smoothing_function=SmoothingFunction().method1)

	def close(self):
		'''Return a dict which contains:

			* **fwbwbleu**: fw/bw bleu value.
		'''
		max_len = max([len(self.hyps), len(self.refs)])
		if self.sample > max_len:
			self.sample = max_len

		random.shuffle(self.hyps)
		random.shuffle(self.refs)
		try:
			result = {}
			for ngram in range(2, 5):
				weight = tuple((1. / ngram for _ in range(ngram)))
				if self.sample >= 1000:
					pool = Pool(multiprocessing.cpu_count())
					bleu_irl_fw = pool.map(self.run_f, \
							[(self.refs, self.hyps[i], weight) for i in range(self.sample)])
					bleu_irl_bw = pool.map(self.run_f, \
							[(self.hyps, self.refs[i], weight) for i in range(self.sample)])
					pool.close()
					pool.join()
				else:
					bleu_irl_fw, bleu_irl_bw = [], []
					for i in range(self.sample):
						bleu_irl_fw.append(self.run_f((self.refs, self.hyps[i], weight)))
						bleu_irl_bw.append(self.run_f((self.hyps, self.refs[i], weight)))

				fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
				bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
				result["fw-bleu-%d"%ngram] = fw_bleu
				result["bw-bleu-%d"%ngram] = bw_bleu
				result["fw-bw-bleu-%d"%ngram] = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
			return result
		except ZeroDivisionError as _:
			raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")


class MultiTurnBleuCorpusMetric(MetricBase):
	'''Metric for calculating multi-turn BLEU.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		reference_allvocabs_key (str): Reference sentences with all vocabs are passed to
			:func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``reference_allvocabs``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
		turn_len_key (str): Turn length are passed to  :func:.forward by
			``data[turn_len_key]``. Default: ``turn_length``.
	'''
	def __init__(self, dataloader, reference_allvocabs_key="reference_allvocabs", \
					gen_key="gen", \
					turn_len_key="turn_length" \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.turn_len_key = turn_len_key
		self.gen_key = gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`):
				Reference sentences with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array`): 3-d array of int.
				Sentences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				The 2nd / 3rd dimension can be jagged.
				Size: `[batch_size, max_turn_length, gen_sentence_length]`.
			data[turn_len_key] (list or :class:`numpy.array`): Length of turns in each sample.
				Size: `[batch_size]`
		'''
		reference_allvocabs = data[self.reference_allvocabs_key]
		length = data[self.turn_len_key]
		gen = data[self.gen_key]
		if len(length) != len(reference_allvocabs) or len(length) != len(gen):
			raise ValueError("Batch num is not matched.")

		for i, turn_length in enumerate(length):
			gen_session = gen[i]
			ref_session = reference_allvocabs[i]
			for j in range(turn_length):
				self.hyps.append(self.dataloader.trim_index(gen_session[j]))
				self.refs.append([self.dataloader.trim_index(ref_session[j])[1:]])

	def close(self):
		'''Return a dict which contains:

			* **bleu**: bleu value.
		'''
		try:
			return {"bleu": \
				corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method7)}
		except ZeroDivisionError as _:
			raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")

class SingleTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		post_allvocabs_key (str): Dialog post are passed to :func:`forward`
			by ``data[post_allvocabs_key]``.
			Default: ``post``.
		resp_allvocabs_key (str): Dialog responses are passed to :func:`forward`
			by ``data[resp_allvocabs_key]``.
			Default: ``resp``.
		gen_key (str): Sentence generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, post_allvocabs_key="post_allvocabs", \
			resp_allvocabs_key="resp_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.post_allvocabs_key = post_allvocabs_key
		self.resp_allvocabs_key = resp_allvocabs_key
		self.gen_key = gen_key
		self.post_list = []
		self.resp_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[post_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog posts with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[resp_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog responses with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				Contains end token (eg: ``<eos>``)`, but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		post_allvocabs = data[self.post_allvocabs_key]
		resp_allvocabs = data[self.resp_allvocabs_key]
		gen = data[self.gen_key]
		if len(post_allvocabs) != len(resp_allvocabs) or len(resp_allvocabs) != len(gen):
			raise ValueError("Batch num is not matched.")
		for i, post_sen in enumerate(post_allvocabs):
			self.post_list.append(self.dataloader.index_to_sen(post_sen[1:]))
			self.resp_list.append(self.dataloader.index_to_sen(resp_allvocabs[i][1:]))
			self.gen_list.append(self.dataloader.index_to_sen(gen[i]))

	def close(self):
		'''Return a dict which contains:

			* **post**: a list of post sentences.
			* **resp**: a list of response sentences.
			* **gen**: a list of generated sentences.
		'''
		return {"post": self.post_list, "resp": self.resp_list, "gen": self.gen_list}

class MultiTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		context_allvocabs_key (str): Dialog context are passed to :func:`forward` by
			``data[context_key]``. Default: ``context_allvocabs``.
		reference_allvocabs_key (str): Dialog references with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``reference_allvocabs``.
		gen_key (str): Sentences generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
		turn_len_key (str): Turn length are passed to :func:.forward by
			``data[turn_len_key]``. Default: ``turn_length``.
	'''
	def __init__(self, dataloader, context_allvocabs_key="context_allvocabs", \
			reference_allvocabs_key="reference_allvocabs", gen_key="gen", \
			turn_len_key="turn_length"):
		super().__init__()
		self.dataloader = dataloader
		self.context_allvocabs_key = context_allvocabs_key
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.turn_len_key = turn_len_key
		self.context_list = []
		self.reference_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[context_allvocabs_key] (list or :class:`numpy.array` of `int`): Dialog post.
				A 3-d padded array containing id of words.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, _turn_length, max_sentence_length]`
			data[reference_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog responses with all vocabs. A 3-d padded array containing id of words.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				A 3-d padded array containing id of words.
				Contains  end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, max_turn_length, gen_sentence_length]`.
			data[turn_len_key] (list or :class:`numpy.array`): Length of turns in each sample.
				Size: `[batch_size]`
		'''
		context_allvocabs = data[self.context_allvocabs_key]
		reference_allvocabs = data[self.reference_allvocabs_key]
		gen = data[self.gen_key]
		turn_length = data[self.turn_len_key]
		if len(gen) != len(reference_allvocabs):
			raise ValueError("Batch num is not matched.")
		for i, context_sen in enumerate(context_allvocabs):
			self.context_list.append(self.dataloader.multi_turn_index_to_sen( \
				np.array(context_sen), ignore_first_token=True))
			self.reference_list.append(self.dataloader.multi_turn_index_to_sen( \
				np.array(reference_allvocabs[i]), turn_length=turn_length[i], ignore_first_token=True))
			self.gen_list.append(self.dataloader.multi_turn_index_to_sen( \
				np.array(gen[i]), turn_length=turn_length[i]))
			print(turn_length[i])
			print(len(self.reference_list[-1]))

			if len(self.reference_list[-1]) != len(self.gen_list[-1]):
				raise ValueError("Reference turn num %d != gen turn num %d." % \
						(len(self.reference_list[-1]), len(self.gen_list[-1])))

	def close(self):
		'''Return a dict which contains:

			* **context**: a list of post sentences.
			* **reference**: a list of response sentences.
			* **gen**: a list of generated sentences.
		'''
		return {"context": self.context_list, "reference": self.reference_list, "gen": self.gen_list}

class LanguageGenerationRecorder(MetricBase):
	'''A metric-like class for recorder BLEU.

	Arguments:
		dataloader (:class:contk.BasicLanguageGeneration): A language generation dataloader.
		gen_key (str): Sentences generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[gen_key] (list or :class:`numpy.array` of `int`): Sentences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		for sen in gen:
			self.gen_list.append(self.dataloader.index_to_sen(sen))

	def close(self):
		'''Return a dict which contains:

			* **gen**: a list of generated sentences.
		'''
		return {"gen": self.gen_list}

class HashValueRecorder(MetricBase):
	'''A metric-like class for recording hash value metric.
	'''
	def __init__(self, hash_key="hashvalue"):
		super().__init__()
		self._hash_key = hash_key
		self.unordered_hash = None

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains hashvalue.
		'''
		if "hashvalue" in data:
			if self.unordered_hash is None:
				self.unordered_hash = UnorderedSha256()
			self.unordered_hash.update_hash(data["hashvalue"])

	def close(self):
		'''Return a dict which contains the items which all the
			metric components returned.
		'''
		if self.unordered_hash:
			return {self._hash_key: self.unordered_hash.digest()}
		else:
			return {}

class MetricChain(MetricBase):
	'''A metric-like class for stacked metric. You can use this class
	making multiples metric combination like one.

	Examples:
		>>> metric = MetricChain()
		>>> metric.add_metric(BleuCorpusMetric())
		>>> metric.add_metric(SingleDialogRecorder(dataloader))
	'''
	def __init__(self):
		super().__init__()
		self.metric_list = []

	def add_metric(self, metric):
		'''Add metric for processing.

		Arguments:
			metric (MetricBase): a metric class
		'''
		if not isinstance(metric, MetricBase):
			raise TypeError("Metric must be a subclass of MetricBase")
		self.metric_list.append(metric)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains keys which all the
				metric components need.
		'''
		for metric in self.metric_list:
			metric.forward(data)

	def close(self):
		'''Return a dict which contains the items which all the
			metric components returned.
		'''
		ret_dict = {}
		for metric in self.metric_list:
			ret_dict.update(metric.close())
		return ret_dict
