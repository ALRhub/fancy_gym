# This conf.py is in large parts inspired by the oen used by stable-baselines 3

import datetime

project = 'Fancy Gym'
author = 'Fabian Otto, Onur Celik, Dominik Roth, Hongyi Zhou'
copyright = f'2020-{datetime.date.today().year}, {author}'

release = '0.2'  # The full version, including alpha/beta/rc tags
version = '0.2'  # The short X.Y version

extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
]

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']

language = "en"
pygments_style = "sphinx"

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_logo = "_static/imgs/icon.svg"
epub_show_urls = 'footnote'
html_static_path = ['_static']

def setup(app):
    app.add_css_file("style.css")