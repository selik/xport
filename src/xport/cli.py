"""
Read and write SAS XPORT/XPT-format files.
"""

# Standard Library
import functools
import json
import logging
import logging.config
import sys

# Community Packages
import click
import yaml

# Xport Modules
import xport
import xport.v56
import xport.v89

__all__ = [
    'cli',
]

try:
    yaml.load = functools.partial(yaml.load, Loader=yaml.CSafeLoader)
except AttributeError:
    yaml.load = functools.partial(yaml.load, Loader=yaml.SafeLoader)

try:
    with open('logging.yml') as file:
        LOG_CONFIG = yaml.load(file)
except FileNotFoundError:
    LOG_CONFIG = {'version': 1}
logging.config.dictConfig(LOG_CONFIG)

LOG = logging.getLogger(__name__)
log_levels = [name for x, name in sorted(logging._levelToName.items()) if x]


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
)
@click.argument('input', type=click.File('rb'))
@click.argument(
    'output',
    type=click.File('wt'),
    default=sys.stdout,
)
@click.option(
    '--dataset', metavar='NAME', help='Select a dataset by name.  Defaults to the first dataset.'
)
@click.option(
    '--loglevel',
    metavar='LEVEL',
    type=click.Choice(log_levels, case_sensitive=False),
    help=f'Set logging level.  {{{", ".join(log_levels[:-1])}}}',
)
@click.version_option(version=xport.__version__)
def cli(input, output, dataset, loglevel):
    """
    Convert SAS Transport (XPORT) files to comma-separated values (CSV).
    """
    if loglevel:
        for k, config in LOG_CONFIG['loggers'].items():
            config['level'] = loglevel.upper()
        logging.config.dictConfig(LOG_CONFIG)

    LOG.debug('Xport version %s', xport.__version__)
    LOG.debug('CLI arg --loglevel = %r', loglevel)
    LOG.debug('Using logging config %s', json.dumps(LOG_CONFIG, indent=2))

    bytestring = input.read()
    if xport.v89.Library.pattern.match(bytestring):
        library = xport.v89.loads(bytestring)
    else:
        library = xport.v56.loads(bytestring)
    if dataset is not None:
        ds = library[dataset]
    elif library:
        ds = next(iter(library.values()))
    else:
        raise ValueError("Library has no member datasets")
    LOG.info(f'Selected dataset {ds.name!r}')
    ds.to_csv(output, index=False)
