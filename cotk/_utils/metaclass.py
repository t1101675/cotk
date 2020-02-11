"""A lib for decorator and metaclass"""
import re
import inspect
import types
from functools import wraps

class DocStringInheritor(type):
	"""
	A meta class. It make the class:

	* Docstring can inherit the parent classes.
	* {STRING_DOCS} in docs will be replaced by self.STRING_DOCS
	* {BaseClassName.STRING_DOCS} in docs will be replaced by BaseClassName.STRING_DOCS
	* If the docstring iherited from the parent class Base, {_STRING_DOCS} use the value of self._STRING_DOCS,
	  but {STRING_DOCS} (without underscore prefix) will use the value of Base._STRING_DOCS
	* The replacement can be nested

	A variation on
	http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
	by Paul McGuire
	from https://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
	"""
	def __new__(cls, name, bases, clsdict):

		def find_base(base_name):
			if name == base_name:
				return clsdict
			for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
				if mro_cls.__name__ == base_name:
					return mro_cls
			raise ValueError("No bases named %s" % base_name)

		def find_attr(attr_name):
			if "." in attr_name:
				base_name, attr_name = attr_name.split(".")
				base = find_base(base_name)
				if isinstance(base, dict):
					return base[attr_name]
				else:
					return getattr(base, attr_name)
			else:
				if attr_name in clsdict:
					return clsdict[attr_name]
				for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
					if hasattr(mro_cls, attr_name):
						return getattr(mro_cls, attr_name)
				raise KeyError("No such attr.")

		def replace_for_clsdict(matched):
			attr_name = matched.group(1)
			try:
				return find_attr(attr_name)
			except ValueError as err:
				if err.args[0].startswith("No bases"):
					raise ValueError("Can't find %s when interpreting docstring of class %s, because the class doesn't have a baseclass named %s." \
						% (attr_name, name, attr_name.split(".")[0]))
				else:
					raise
			except (AttributeError, KeyError):
				raise ValueError("Can't find %s when interpreting docstring of class %s, please check whether the CONSTANT exists." \
					% (attr_name, name))

		def replace_for(attr):
			def replace(matched):
				attr_name = matched.group(1)
				try:
					return find_attr(attr_name)
				except ValueError as err:
					if err.args[0].startswith("No bases"):
						raise ValueError("Can't find %s when interpreting docstring of %s.%s, because the class doesn't have a baseclass named %s." \
							% (attr_name, name, attr, attr_name.split(".")[0]))
					else:
						raise
				except (AttributeError, KeyError):
					raise ValueError("Can't find %s when interpreting docstring of %s.%s, please check whether the CONSTANT exists." \
						% (attr_name, name, attr))
			return replace

		# modify class docstring
		# first, inherit docstring from bases
		if not('__doc__' in clsdict and clsdict['__doc__']):
			for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
				# iterate from bases in MRO
				if mro_cls.META_DOC:
					clsdict['__doc__'] = mro_cls.META_DOC
					break
		# second, subtitute CONSTANT with _ prefix
		while True:
			doc = re.sub(r'\{\b((\w*\.)?[A-Z][A-Z_]*?)\}', replace_for_clsdict, clsdict['__doc__'])
			if doc == clsdict['__doc__']:
				break
			clsdict['__doc__'] = doc
		# save docstring now as META_DOC, before the final subtitution
		clsdict['META_DOC'] = clsdict['__doc__']
		# final, do substitution for CONSTANT
		while True:
			doc = re.sub(r'\{\b((\w*\.)?[A-Z_]+?)\}', replace_for_clsdict, clsdict['__doc__'])
			if doc == clsdict['__doc__']:
				break
			clsdict['__doc__'] = doc

		meta_doc = clsdict['META_DOC_FOR_ATTRIBUTES'] = {}

		# modify attribute docstring
		for attr, attribute in clsdict.items():

			doc = attribute.__doc__
			# first, inherit docstring from bases
			if not doc:
				for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()
								if hasattr(mro_cls, attr)):
					if attr in getattr(mro_cls, "META_DOC_FOR_ATTRIBUTES", {}):
						doc = mro_cls.META_DOC_FOR_ATTRIBUTES[attr]
						break
				else:
					doc = ""

			# second, subtitute CONSTANT with _ prefix
			while True:
				# else do substitution for CONSTANT
				new_doc = re.sub(r'\{\b(((\w*\.)?)[A-Z][A-Z_]*?)\}', replace_for(attr), doc)
				if doc == new_doc:
					break
				doc = new_doc

			# save docstring now as META_DOC, before the final subtitution
			meta_doc[attr] = doc

			# final, do substitution for CONSTANT
			while True:
				# else do substitution for CONSTANT
				new_doc = re.sub(r'\{\b(((\w*\.)?)[A-Z_]+?)\}', replace_for(attr), doc)
				if doc == new_doc:
					break
				doc = new_doc

			# set all doc
			if not doc:
				doc = None
			if doc == attribute.__doc__:
				continue
			elif isinstance(attribute, property):
				clsdict[attr] = property(
						attribute.fget, attribute.fset, attribute.fdel, doc) # type: ignore
			else:
				attribute.__doc__ = doc

		return type.__new__(cls, name, bases, clsdict)

class LoadClassInterface:
	r"""The support of dynamic class load."""
	@classmethod
	def get_all_subclasses(cls):
		'''Return a generator of all subclasses.

		Returns:
			(generator) A iterator over all subclasses.
		'''
		for subclass in cls.__subclasses__():
			yield from subclass.get_all_subclasses()
			yield subclass

	@classmethod
	def load_class(cls, class_name):
		'''Return a subclass of ``class_name``, case insensitively.

		Arguments:
			class_name (str): target class name.

		Returns:
			(class) The subclass specified by ``class_name``
		'''
		result = None
		for subclass in cls.get_all_subclasses():
			if subclass.__name__.lower() == class_name.lower():
				if result is None:
					result = subclass
				else:
					raise RuntimeError('There are two classes with the name "{}" located at "{}" and "{}". \
						You have to remove one of them to make "load_class" work normally.'.format(\
						class_name, inspect.getfile(result), inspect.getfile(subclass)))
		return result

def copy_func(target, cls, method_name):
	method = getattr(cls, method_name)
	assert callable(method)
	@wraps(method)
	def new_method(self, *args, **kwargs):
		return method(target(self), *args, **kwargs)
	new_method.__doc__ = cls.META_DOC_FOR_ATTRIBUTES[method_name] # type: ignore
	return new_method

def copy_property(target, cls, old_property_name):
	old_property = getattr(cls, old_property_name)
	assert isinstance(old_property, property)
	def new_property_fget(self):
		return old_property.fget(target(self))
	def new_property_fset(self, a):
		return old_property.fset(target(self, a))
	def new_property_fdel(self):
		return old_property.fdel(target(self))
	doc = cls.META_DOC_FOR_ATTRIBUTES[old_property_name]
	return property(new_property_fget, new_property_fset, new_property_fdel, doc) #type: ignore
