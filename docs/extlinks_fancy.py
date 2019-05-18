# -*- coding: utf-8 -*-
"""
    Extension to save typing and prevent hard-coding of base URLs in the reST
    files.

    This adds a new config value called ``extlinks_fancy`` that is created like this::

       extlinks_fancy = {'exmpl': ('http://example.com/{0}.html', "Example {0}"), ...}

    You can also give an explicit caption, e.g. :exmpl:`Foo <foo>`.

    :copyright: Copyright 2007-2017 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from six import iteritems
from docutils import nodes, utils

import sphinx
from sphinx.util.nodes import split_explicit_title


def make_link_role(base_urls, prefixes):
    def role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
        text = utils.unescape(text)
        has_explicit_title, title, part = split_explicit_title(text)
        pnodes = []
        for base_url, prefix in zip(base_urls, prefixes):
            full_url = base_url.format(part)
            if not has_explicit_title:
                if prefix is None:
                    title = full_url
                else:
                    title = prefix.format(part)
            ref = nodes.reference(title, title, internal=False, refuri=full_url)
            if len(pnodes) == 1:
                pnodes.append(nodes.Text(" ["))
            if len(pnodes) > 2:
                pnodes.append(nodes.Text(", "))
            pnodes.append(ref)
        if len(base_urls) > 1:
            pnodes.append(nodes.Text("]"))
        return pnodes, []
    return role


def setup_link_roles(app):
    for name, (base_urls, prefixes) in iteritems(app.config.extlinks_fancy):
        app.add_role(name, make_link_role(base_urls, prefixes))


def setup(app):
    app.add_config_value('extlinks_fancy', {}, 'env')
    app.connect('builder-inited', setup_link_roles)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}

