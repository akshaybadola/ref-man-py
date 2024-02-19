from typing import Optional
from abc import ABC, abstractmethod, abstractproperty
import logging
from pathlib import Path
import os
import shutil
from subprocess import Popen, PIPE, TimeoutExpired
from threading import Thread, Event

from common_pyutil.sqlite import SQLite
from common_pyutil.monitor import Timer


class Backend(ABC):
    def __init__(self, cache_file):
        self._cache_file = cache_file
        self._cache = {}

    def backup(self):
        shutil.copyfile(str(self._cache_file), str(self._cache_file) + ".bak")

    @abstractmethod
    def refresh(self):
        pass

    @abstractmethod
    def get(self, key) -> Optional[str]:
        pass

    @abstractmethod
    def add(self, key, val):
        pass

    @abstractmethod
    def delete(self, key):
        pass

    @abstractmethod
    def delete_many(self, key):
        pass

    @abstractmethod
    def insert_many(self, keyvals: list[tuple[str, str]]):
        pass

    @property
    def broken_links(self) -> list[str]:
        return [c[0] for c in self._cache.items() if c[1] == ""]

    @property
    def data(self) -> list[tuple[str, str]]:
        return [*self._cache.items()]

    @property
    def as_lists(self) -> tuple[list[str], list[str]]:
        return [*self._cache.keys()], [*self._cache.values()]


class TextBackend(Backend):
    def __init__(self, cache_file, logger):
        super().__init__(cache_file)
        self.logger = logger
        self.refresh()

    def _write_cache(self):
        with open(self._cache_file, "w") as f:
            f.write("\n".join(";".join(x) for x in self._cache.items()))

    def refresh(self):
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                cache = [x for x in f.read().split("\n") if len(x)]
        else:
            self.logger.error(f"Cache file {self._cache_file} does not exist")
            cache = []
        self._cache = dict(x.split(";") for x in cache)

    def get(self, fname) -> Optional[str]:
        return self._cache.get(fname, None)

    def add(self, fname, link):
        self._cache[fname] = link
        with open(self._cache_file, "a") as f:
            f.write(f"{fname};{link}\n")

    def delete(self, fname):
        self._cache.pop(fname)
        self._write_cache()

    def delete_many(self, fnames):
        for dl in fnames:
            self._cache.pop(dl)
        self._write_cache()

    def insert_many(self, links: list[tuple[str, str]]):
        with open(self._cache_file, "a") as f:
            f.write("\n".join(";".join(x) for x in links))


class SQLiteBackend(Backend):
    def __init__(self, dbfile, logger):
        super().__init__(dbfile)
        self.db = str(Path(dbfile).name)
        self.table = "links"
        self.sqlite = SQLite(Path(dbfile).parent, logger.name)
        self._cache: dict[str, str] = {}
        self.sqlite.create_table(self.db, self.table, ["filename", "link"], "filename")
        self.refresh()

    def refresh(self):
        result, column_names = self.sqlite.select_data(self.db, self.table)
        self._cache = dict(result)

    def get(self, fname) -> Optional[str]:
        return self._cache.get(fname, None)

    def add(self, fname, link):
        if fname in self._cache:
            self.sqlite.update_data(self.db, self.table,
                                    {"filename": fname, "link": link},
                                    "filename", fname)
        else:
            self.sqlite.insert_data(self.db, self.table,
                                    {"filename": fname, "link": link})
        self._cache[fname] = link

    def delete(self, fname):
        self._cache.pop(fname)
        self.sqlite.delete_rows(self.db, self.table, f"filename == '{fname}'")

    def delete_many(self, fnames):
        for fname in fnames:
            self._cache.pop(fname)
        self.sqlite.delete_many_rows(self.db, self.table,
                                     [f"filename == '{fname}'" for fname in fnames])

    def insert_many(self, links: list[tuple[str, str]]):
        self.sqlite.insert_many(self.db, self.table,
                                [{"filename": k, "link": v} for k, v in links
                                 if k not in self._cache])


class PDFCache:
    """A local and remote (via :code:`rclone`) PDF files manager

       The pdf files are linked to publications and be stored in any `rclone`
       remote instance. The local links are stored in the :code:`cache_file`
       and can be updated on command.

       Args:
           local_dir: Local dirctory where pdf files are stored
           remote_dir: :code:`rclone` remote dirctory where pdf files are stored
           cache_file: ';' separated file of pdf links
           logger: the logger instance

    """
    def __init__(self, local_dir: Path, remote_dir: Path,
                 cache_file: Path, logger: logging.Logger):
        self.timer = Timer()
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.cache_file = cache_file
        self.updating_ev = Event()
        self.success_ev = Event()
        self.success_with_errors_ev = Event()
        self.update_thread: Optional[Thread] = None
        self.logger = logger
        # self.backend = TextBackend(self.cache_file, self.logger)
        self.backend = SQLiteBackend(self.cache_file, self.logger)
        self.check_and_fix_cache()

    @property
    def updating(self) -> bool:
        return self.updating_ev.is_set()

    @property
    def finished(self) -> bool:
        return self.success_ev.is_set()

    @property
    def finished_with_errors(self) -> bool:
        return self.success_with_errors_ev.is_set()

    @property
    def cache_needs_updating(self) -> set:
        local_files = self.local_files
        self.backend.refresh()
        cache, cache_files = self.backend.data
        files = set(local_files) - set(cache_files)
        return files

    @property
    def local_files(self):
        return [os.path.join(self.local_dir, f)
                for f in os.listdir(self.local_dir)
                if not f.startswith(".")]

    def read_cache(self) -> tuple[list[str], list[str], list[str]]:
        self.backend.refresh()
        return self.local_files, *self.backend.as_lists

    def _remote_path(self, fname) -> str:
        return os.path.join(self.remote_dir, os.path.basename(fname))

    def _local_path(self, fname: str) -> str:
        return os.path.join(self.local_dir, os.path.basename(fname))

    def stop_update(self) -> None:
        self.updating_ev.clear()

    def shutdown(self) -> None:
        self.stop_update()
        if self.update_thread is not None:
            self.update_thread.join()

    def check_and_fix_cache(self) -> None:
        """Delete files from cache which have been deleted on disk and start update thread
        if any links are missing.

        """
        self.logger.debug("Checking existing cache")
        local_files, fnames, remote_files = self.read_cache()
        self.logger.debug(f"We have {len(local_files)} pdf files, {len(fnames)} entries in cache" +
                          f" and {len(remote_files)} remote files")
        deleted_files = set(fnames) - set(local_files)
        if deleted_files:
            self.logger.info(f"Files {deleted_files} not on disk. Removing from cache.")
            self.backend.delete_many(deleted_files)
        else:
            self.logger.debug("No deleted links")
        broken_links = self.backend.broken_links
        if broken_links:
            self.logger.debug(f"Found {len(broken_links)} broken links. Updating")
            self.update_thread = Thread(target=self.update_cache_helper, args=[broken_links])
            self.update_thread.start()
        else:
            self.logger.debug("No broken links")

    def copy_file_from_remote(self, fname: str) -> bool:
        """Copy file from local to remote.

        Args:
            fname: filename


        """
        remote_path = self._remote_path(fname)
        try:
            p = Popen(f"rclone --no-update-modtime -v copy {remote_path} {self.local_dir}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate(timeout=10)
            err = err.decode("utf-8").lower()  # type: ignore
            if err and ("copied" in err or "transferred" in err):  # type: ignore
                self.logger.debug(f"Copied file {remote_path} to local")
                status = True
            else:
                status = False
        except TimeoutExpired:
            self.logger.warning(f"Timeout while copying for file {remote_path}")
            status = False
        return status

    def copy_file_to_remote(self, fname: str) -> bool:
        """Copy file from local to remote.

        Args:
            fname: filename


        """
        local_path = self._local_path(fname)
        try:
            p = Popen(f"rclone --no-update-modtime -v copy {local_path} {self.remote_dir}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate(timeout=10)
            err = err.decode("utf-8").lower()  # type: ignore
            if err and ("copied" in err or "transferred" in err):  # type: ignore
                self.logger.debug(f"Copied file {local_path} to remote")
                status = True
            else:
                status = False
        except TimeoutExpired:
            self.logger.warning(f"Timeout while copying for file {local_path}")
            status = False
        return status

    def try_get_link(self, remote_path: str) -> tuple[bool, str]:
        """Try and fetch a shareable link for an :code:`rclone` remote_path.

        :code:`rclone` remote paths are prepended with a remote :code:`name` so the path is
        :code:`name:path`. Depending on the remote the status and error messages may
        differ. Currently, these messages are in :code:`gdrive` format.

        Args:
            remote_path: Remote path for which to fetch the link

        """
        self.logger.debug(f"Fetching link for {remote_path}")
        try:
            p = Popen(f"rclone -v link {remote_path}", shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate(timeout=10)
            if err:
                if "error 403" in err.decode().lower() or\
                   "object not found" in err.decode().lower():
                    status = False
                    link = "NOT_PRESENT"
                else:
                    status = False
                    link = f"OTHER_ERROR. {err.decode('utf-8')}"
            else:
                link = out.decode("utf-8").replace("\n", "")
                if link:
                    status = True
                else:
                    status = False
                    link = "EMPTY_RESPONSE"
        except TimeoutExpired:
            self.logger.warning(f"Timeout while getting link for file {remote_path}")
            link = "TIMEOUT"
            status = False
        return status, link

    def get_link(self, fname: str, warnings: list[str], no_update: bool = False) -> Optional[str]:
        """Get a link for an file name.

        Copy the file to the remote path if it doesn't exist there.

        This function tries to get the link TWICE in case an error occurs first
        time.

        Args:
            fname: Local filename for which to fetch the link
            warnings: A shared variable where warnings are appened if any occur
            no_update: Flag to not update the cache at once in case one wishes to
                       collect links and write them in backed all at once later.

        """
        maybe_link = self.backend.get(fname)
        if maybe_link:
            return maybe_link
        try:
            with self.timer:
                remote_path = self._remote_path(fname)
                if " " in remote_path:
                    remote_path = f'"{remote_path}"'
                status, link = self.try_get_link(remote_path)
                if not status:
                    if link == "NOT_PRESENT":
                        self.logger.warning(f"File {fname} does not exist on remote. Copying")
                        status = self.copy_file_to_remote(fname)
                        if status:
                            status, link = self.try_get_link(remote_path)
                    else:
                        raise ValueError(f"Error {link} for {remote_path}")
            if not status:
                warnings.append(f"{fname}")
                self.logger.error(f"Error occurred for file {fname} {link}")
                return None
            else:
                self.logger.debug(f"got link {link} for file {fname} in {self.timer.time} seconds")
                if not no_update:
                    self.backend.add(fname, link)
                return link
        except Exception as e:
            self.logger.error(f"Error occured for file {fname} {e}")
        return None

    def update_cache(self, at_end=True) -> None:
        """Update the local cache

        For each file on the local machine fetch a shareable link from the remote dir.

        """
        if not self.updating:
            self.update_thread = Thread(target=self.update_cache_helper, kwargs={"at_end": at_end})
            self.update_thread.start()
        else:
            self.logger.error("We are still updating")

    def sync_remote_dir_to_local(self) -> None:
        """Equivalent to :code:`rclone sync REMOTE LOCAL`

        WILL DELETE files in local dir. Use :meth:`copy_remote_dir_to_local` to
        avoid deletion

        """
        self.logger.debug(f"Syncing remote {self.remote_dir} to {self.local_dir}")
        try:
            p = Popen(f"rclone -v sync {self.remote_dir} {self.local_dir}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()
        except Exception as e:
            self.logger.error(f"Error occured {e}")

    def copy_local_dir_to_remote(self) -> None:
        """Equivalent to :code:`rclone copy REMOTE LOCAL`

        Does NOT DELETE files in local dir and only copies new files.

        """
        self.logger.debug(f"Syncing remote {self.local_dir} to {self.remote_dir}")
        try:
            p = Popen(f"rclone -v copy --no-update-modtime {self.remote_dir} {self.local_dir}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()
        except Exception as e:
            self.logger.error(f"Error occured {e}")

    def copy_remote_dir_to_local(self) -> None:
        """Equivalent to :code:`rclone copy REMOTE LOCAL`

        Does NOT DELETE files in local dir and only copies new files.

        """
        self.logger.debug(f"Syncing remote {self.remote_dir} to {self.local_dir}")
        try:
            p = Popen(f"rclone -v copy --no-update-modtime {self.remote_dir} {self.local_dir}",
                      shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()
        except Exception as e:
            self.logger.error(f"Error occured {e}")

    def update_cache_helper(self, fix_files: list[str] = [], at_end: bool = True) -> None:
        """Update links for files in local_dir if they're missing

        This function gets missing links all at once and then writes it to the backend.

        Args:
            fix_files: Files for which links are missing


        """
        if not self.updating_ev.is_set():
            self.updating_ev.set()
        if self.success_ev.is_set():
            self.success_ev.clear()
        if self.success_with_errors_ev.is_set():
            self.success_with_errors_ev.clear()
        self.logger.info(f"Updating local cache {self.cache_file}")
        try:
            warnings: list[str] = []
            local_files, cache, remote_files = self.read_cache()
            files = list(set(local_files) - set(remote_files))
            if fix_files:
                for f in fix_files:
                    if f in cache:
                        self.backend.delete(f)
                files = fix_files
            init_cache_size = len(cache)
            cache_dict = dict(self.backend.data)
            self.logger.info(f"Will try to fetch links for {len(files)} files")
            links = []
            for f in files:
                if not self.updating_ev.is_set():
                    break
                link = self.get_link(f, warnings, not at_end)
                if link:
                    links.append((f, link))
            if at_end:
                self.logger.info(f"Writing {len(cache_dict) - init_cache_size} links to {self.cache_file}")
                self.backend.backup()
                self.backend.insert_many(links)
            self.updating_ev.clear()
            if warnings:
                self.success_with_errors_ev.set()
            else:
                self.success_ev.set()
        except Exception as e:
            self.updating_ev.clear()
            self.logger.error(f"Error {e} while updating cache")
            self.logger.error(f"Overwritten {self.cache_file}.\n" +
                              f"Original file backed up to {self.cache_file}.bak")
