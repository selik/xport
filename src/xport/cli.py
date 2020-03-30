"""
Read and write SAS XPORT/XPT-format files.
"""

# Community Packages
import click

# Xport Modules
import xport

__all__ = [
    'cli',
]


def print_version(ctx, param, value):
    """
    Echo the program version information.
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'xport {xport.__version__}')
    ctx.exit()


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
)
@click.option(
    '--version',
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
)
def cli():
    """
    Read and write SAS XPORT/XPT-format files.
    """
