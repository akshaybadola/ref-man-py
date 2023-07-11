import os
import sys
import argparse
from pathlib import Path


from . import __version__


def main():
    default_config_dir = Path.home().joinpath(".config/ref-man")
    parser = argparse.ArgumentParser("ref-man", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--no-threaded", dest="threaded", action="store_false",
                        help="Whether flask server should be threaded or not")
    parser.add_argument("--host", default="localhost",
                        help="host on which to bind the python server")
    parser.add_argument("--port", "-p", type=int, default=9999,
                        help="Port to bind to the python server")
    parser.add_argument("--proxy-port", dest="proxy_port", type=int, default=0,
                        help="HTTP proxy server port for method 'fetch_proxy'")
    parser.add_argument("--proxy-everything", dest="proxy_everything", action="store_true",
                        help="Should we proxy all requests?")
    parser.add_argument("--proxy-everything-port", dest="proxy_everything_port",
                        type=int, default=0,
                        help="HTTP proxy server port if proxy_everything is given")
    parser.add_argument("--data-dir", "-d", dest="data_dir", type=Path,
                        default=default_config_dir.joinpath("s2_cache"),
                        help="Semantic Scholar cache directory")
    parser.add_argument("--refs-cache-dir", "-r", dest="refs_cache_dir", type=Path,
                        help="Semantic Scholar References cache directory\n" +
                        "It's built from full dump of Semantic Scholar data")
    parser.add_argument("--local-pdfs-dir", "-l", dest="local_pdfs_dir", type=Path,
                        default=default_config_dir.joinpath("pdfs"),
                        help="Local directory where pdfs are stored")
    parser.add_argument("--remote-pdfs-dir", dest="remote_pdfs_dir", type=Path,
                        default="", help="Remote rclone pdfs directory")
    parser.add_argument("--remote-links-cache", dest="remote_links_cache", type=Path,
                        default=Path(""), help="Remote links cache file")
    parser.add_argument("--config-dir", dest="config_dir", type=Path,
                        default=default_config_dir,
                        help="Directory where configuration related files are kept")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=16,
                        help="Simultaneous connections to DBLP")
    parser.add_argument("--chrome-debugger-path", dest="chrome_debugger_path", type=Path,
                        default="",
                        help="Path to chrome debugger script which can validate " +
                        "Semantic Scholar Search params (optional)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--logfile", help="Logfile for logging")
    parser.add_argument("--logdir", type=Path, help="Directory where logfile will be stored")
    parser.add_argument("--logfile-verbosity", type=str, default="debug",
                        choices=["error", "info", "debug"],
                        help="Verbosity level for logfile if given")
    parser.add_argument("--verbosity", "-v", type=str, default="info",
                        choices=["error", "info", "debug"], help="Verbosity level for stdout")
    parser.add_argument("--version", action="store_true",
                        help="Print version and exit.")
    args = parser.parse_args()
    if args.version:
        print(f"ref-man-server version {__version__}")
        sys.exit(0)
    from .service import RefMan
    kwargs = {
        "host": args.host,
        "port": args.port,
        "proxy_port": args.proxy_port,
        "proxy_everything": args.proxy_everything,
        "proxy_everything_port": args.proxy_everything_port,
        "data_dir": args.data_dir,
        "refs_cache_dir": args.refs_cache_dir,
        "config_dir": Path(args.config_dir),
        "local_pdfs_dir": args.local_pdfs_dir,
        "remote_pdfs_dir": args.remote_pdfs_dir,
        "remote_links_cache": args.remote_links_cache,
        "batch_size": args.batch_size,
        "chrome_debugger_path": args.chrome_debugger_path,
        "debug": args.debug,
        "logdir": args.logdir,
        "logfile": args.logfile,
        "logfile_verbosity": args.logfile_verbosity,
        "verbosity": args.verbosity,
        "threaded": args.threaded,
    }
    service = RefMan(**kwargs)
    service.run()


if __name__ == '__main__':
    main()
