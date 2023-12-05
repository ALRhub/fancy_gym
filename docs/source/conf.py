project = 'Fancy Gym'
copyright = '2023, Fabian Otto, Onur Celik'
author = 'Fabian Otto, Onur Celik'

release = '0.4'  # The full version, including alpha/beta/rc tags
version = '0.4'  # The short X.Y version

extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'
epub_show_urls = 'footnote'
html_static_path = ['_static']
