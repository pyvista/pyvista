# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Backreferences Generator
========================

Parses example file code in order to keep track of used functions
"""
from __future__ import print_function, unicode_literals

import ast
import codecs
import collections
import os
import re

from . import sphinx_compatibility
from .scrapers import _find_image_ext
from .utils import _replace_md5

# Try Python 2 first, otherwise load from Python 3
try:
    import cPickle as pickle
except ImportError:
    import pickle
# Try Python 3 first, otherwise load from Python 2
try:
    from html import escape
except ImportError:
    from functools import partial
    from xml.sax.saxutils import escape

    escape = partial(escape, entities={'"': '&quot;'})

from .py_source_parser import parse_source_file, split_code_and_text_blocks


class NameFinder(ast.NodeVisitor):
    """Finds the longest form of variable names and their imports in code

    Only retains names from imported modules.
    """

    def __init__(self):
        super(NameFinder, self).__init__()
        self.imported_names = {}
        self.accessed_names = set()

    def visit_Import(self, node, prefix=''):
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imported_names[local_name] = prefix + alias.name

    def visit_ImportFrom(self, node):
        self.visit_Import(node, node.module + '.')

    def visit_Name(self, node):
        self.accessed_names.add(node.id)

    def visit_Attribute(self, node):
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            # This is a.b, not e.g. a().b
            attrs.append(node.id)
            self.accessed_names.add('.'.join(reversed(attrs)))
        else:
            # need to get a in a().b
            self.visit(node)

    def get_mapping(self):
        for name in self.accessed_names:
            local_name = name.split('.', 1)[0]
            remainder = name[len(local_name):]
            if local_name in self.imported_names:
                # Join import path to relative path
                full_name = self.imported_names[local_name] + remainder
                yield name, full_name


def get_short_module_name(module_name, obj_name):
    """ Get the shortest possible module name """
    scope = {}
    try:
        # Find out what the real object is supposed to be.
        exec('from %s import %s' % (module_name, obj_name), scope, scope)
        real_obj = scope[obj_name]
    except Exception:
        return module_name

    parts = module_name.split('.')
    short_name = module_name
    for i in range(len(parts) - 1, 0, -1):
        short_name = '.'.join(parts[:i])
        scope = {}
        try:
            exec('from %s import %s' % (short_name, obj_name), scope, scope)
            # Ensure shortened object is the same as what we expect.
            assert real_obj is scope[obj_name]
        except Exception:  # libraries can throw all sorts of exceptions...
            # get the last working module name
            short_name = '.'.join(parts[:(i + 1)])
            break
    return short_name


def extract_object_names_from_docs(filename):
    """Add matches from the text blocks (must be full names!)"""
    text = split_code_and_text_blocks(filename)[1]
    text = '\n'.join(t[1] for t in text if t[0] == 'text')
    regex = re.compile(r':(?:'
                       r'func(?:tion)?|'
                       r'meth(?:od)?|'
                       r'attr(?:ibute)?|'
                       r'obj(?:ect)?|'
                       r'class):`(\S*)`'
                       )
    return [(x, x) for x in re.findall(regex, text)]


def identify_names(filename):
    """Builds a codeobj summary by identifying and resolving used names."""
    node, _ = parse_source_file(filename)
    if node is None:
        return {}

    # Get matches from the code (AST)
    finder = NameFinder()
    finder.visit(node)
    names = list(finder.get_mapping())
    names += extract_object_names_from_docs(filename)

    example_code_obj = collections.OrderedDict()
    for name, full_name in names:
        if name in example_code_obj:
            continue  # if someone puts it in the docstring and code
        # name is as written in file (e.g. np.asarray)
        # full_name includes resolved import path (e.g. numpy.asarray)
        splitted = full_name.rsplit('.', 1)
        if len(splitted) == 1:
            # module without attribute. This is not useful for
            # backreferences
            continue

        module, attribute = splitted
        # get shortened module name
        module_short = get_short_module_name(module, attribute)
        cobj = {'name': attribute, 'module': module,
                'module_short': module_short}
        example_code_obj[name] = cobj
    return example_code_obj


def scan_used_functions(example_file, gallery_conf):
    """save variables so we can later add links to the documentation"""
    example_code_obj = identify_names(example_file)
    if example_code_obj:
        codeobj_fname = example_file[:-3] + '_codeobj.pickle.new'
        with open(codeobj_fname, 'wb') as fid:
            pickle.dump(example_code_obj, fid, pickle.HIGHEST_PROTOCOL)
        _replace_md5(codeobj_fname)

    backrefs = set('{module_short}.{name}'.format(**entry)
                   for entry in example_code_obj.values()
                   if entry['module'].startswith(gallery_conf['doc_module']))

    return backrefs


THUMBNAIL_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="{snippet}">

.. only:: html

    .. figure:: /{thumbnail}

        :ref:`sphx_glr_{ref_name}`

.. raw:: html

    </div>
"""

BACKREF_THUMBNAIL_TEMPLATE = THUMBNAIL_TEMPLATE + """
.. only:: not html

    * :ref:`sphx_glr_{ref_name}`
"""


def _thumbnail_div(target_dir, src_dir, fname, snippet, is_backref=False,
                   check=True):
    """Generate RST to place a thumbnail in a gallery."""
    thumb, _ = _find_image_ext(
        os.path.join(target_dir, 'images', 'thumb',
                     'sphx_glr_%s_thumb.png' % fname[:-3]))
    if check and not os.path.isfile(thumb):
        # This means we have done something wrong in creating our thumbnail!
        raise RuntimeError('Could not find internal sphinx-gallery thumbnail '
                           'file:\n%s' % (thumb,))
    thumb = os.path.relpath(thumb, src_dir)
    full_dir = os.path.relpath(target_dir, src_dir)

    # Inside rst files forward slash defines paths
    thumb = thumb.replace(os.sep, "/")

    ref_name = os.path.join(full_dir, fname).replace(os.path.sep, '_')

    template = BACKREF_THUMBNAIL_TEMPLATE if is_backref else THUMBNAIL_TEMPLATE
    return template.format(snippet=escape(snippet),
                           thumbnail=thumb, ref_name=ref_name)


def write_backreferences(seen_backrefs, gallery_conf,
                         target_dir, fname, snippet):
    """Writes down back reference files, which include a thumbnail list
    of examples using a certain module"""
    if gallery_conf['backreferences_dir'] is None:
        return

    example_file = os.path.join(target_dir, fname)
    backrefs = scan_used_functions(example_file, gallery_conf)
    for backref in backrefs:
        include_path = os.path.join(gallery_conf['src_dir'],
                                    gallery_conf['backreferences_dir'],
                                    '%s.examples.new' % backref)
        seen = backref in seen_backrefs
        with codecs.open(include_path, 'a' if seen else 'w',
                         encoding='utf-8') as ex_file:
            if not seen:
                heading = '\n\nExamples using ``%s``' % backref
                ex_file.write(heading + '\n')
                ex_file.write('^' * len(heading) + '\n')
            ex_file.write(_thumbnail_div(target_dir, gallery_conf['src_dir'],
                                         fname, snippet, is_backref=True))
            seen_backrefs.add(backref)


def finalize_backreferences(seen_backrefs, gallery_conf):
    """Replace backref files only if necessary."""
    logger = sphinx_compatibility.getLogger('sphinx-gallery')
    if gallery_conf['backreferences_dir'] is None:
        return

    for backref in seen_backrefs:
        path = os.path.join(gallery_conf['src_dir'],
                            gallery_conf['backreferences_dir'],
                            '%s.examples.new' % backref)
        if os.path.isfile(path):
            _replace_md5(path)
        else:
            level = gallery_conf['log_level'].get('backreference_missing',
                                                  'warning')
            func = getattr(logger, level)
            func('Could not find backreferences file: %s' % (path,))
            func('The backreferences are likely to be erroneous '
                 'due to file system case insensitivity.')
