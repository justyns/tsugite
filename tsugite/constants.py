"""Constants for Tsugite CLI."""

TSUGITE_LOGO_NARROW = """
╔╦╗╔═╗╦ ╦╔═╗╦╔╦╗╔═╗
 ║ ╚═╗║ ║║ ╦║ ║ ║╣
 ╩ ╚═╝╚═╝╚═╝╩ ╩ ╚═╝
"""

TSUGITE_LOGO_WIDE = """
 ███████████                              ███   █████
░█░░░███░░░█                             ░░░   ░░███
░   ░███  ░   █████  █████ ████  ███████ ████  ███████    ██████
    ░███     ███░░  ░░███ ░███  ███░░███░░███ ░░░███░    ███░░███
    ░███    ░░█████  ░███ ░███ ░███ ░███ ░███   ░███    ░███████
    ░███     ░░░░███ ░███ ░███ ░███ ░███ ░███   ░███ ███░███░░░
    █████    ██████  ░░████████░░███████ █████  ░░█████ ░░██████
   ░░░░░    ░░░░░░    ░░░░░░░░  ░░░░░███░░░░░    ░░░░░   ░░░░░░
                                ███ ░███
                               ░░██████          Tsugite
                                ░░░░░░
"""

# The daemon's default address. Kept in core so the server's bind default and the
# client's connect default cannot drift apart.
DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 8374
