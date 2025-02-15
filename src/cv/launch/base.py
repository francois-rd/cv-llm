from collections import namedtuple
import logging

import coma

from ..io import PathConfig, logging as log


# Links a unique config ID with its data type.
ConfigData = namedtuple("ConfigData", "id_ type_")


class Configs:
    """Registry for all known configs."""

    paths = ConfigData("paths", PathConfig)

    @staticmethod
    def add(*cfgs_data: ConfigData):
        """Converts the given config data to a valid coma config dict."""
        return {cfg.id_: cfg.type_ for cfg in cfgs_data}


@coma.hooks.hook
def pre_config_hook(known_args):
    """This pre-config hook set the global default logging level."""
    log.DEFAULT_LEVEL = getattr(logging, known_args.log_level.upper())


@coma.hooks.hook
def pre_run_hook(known_args):
    """This pre-run hook exists early. Useful for debugging init hooks."""
    if known_args.dry_run:
        print("Dry run.")
        quit()


def init():
    """Initiate Coma with application-specific non-default hooks and configs."""
    # ===== 1. Create any additional hooks. =====

    # Parser hook for flagging a dry run.
    dry_run_hook = coma.hooks.parser_hook.factory(
        "--dry-run",
        action="store_true",
        help="exit during pre-run",
    )
    logging_level_hook = coma.hooks.parser_hook.factory(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="set the default global log level",
    )

    # ===== 2. Initialize. =====

    coma.initiate(
        # Add the dry run parser hook.
        parser_hook=coma.hooks.sequence(
            coma.hooks.parser_hook.default,
            dry_run_hook,
            logging_level_hook,
        ),
        # Add the logging level hook.
        pre_config_hook=pre_config_hook,
        # Override the default CLI config_id-to-config_field separator from ':' to '::'.
        # This allows config_field to contain ':' (e.g., a dictionary).
        post_config_hook=coma.hooks.post_config_hook.multi_cli_override_factory(
            coma.config.cli.override_factory(sep="::"),
        ),
        # Add the dry run hook.
        pre_run_hook=pre_run_hook,
    )
