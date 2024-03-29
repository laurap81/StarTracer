# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StarTracer'
copyright = '2024, Laura Posch'
author = 'Laura Posch'
release = '31.01.2024'

# -- Add package to path -----------------------------------------------------

import os
import sys
# import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../StarTracer/'))
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
