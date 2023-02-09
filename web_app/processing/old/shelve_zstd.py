#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manage shelves of pickled objects...now with compression!

A "shelf" is a persistent, dictionary-like object.  The difference
with dbm databases is that the values (not the keys!) in a shelf can
be essentially arbitrary Python objects -- anything that the "pickle"
module can handle.  This includes most class instances, recursive data
types, and objects containing lots of shared sub-objects.  The keys
are ordinary strings.

To summarize the interface (key is a string, data is an arbitrary
object):

        import shelve
        d = shelve.open(filename) # open, with (g)dbm filename -- no suffix

        d[key] = data   # store data at key (overwrites old data if
                        # using an existing key)
        data = d[key]   # retrieve a COPY of the data at key (raise
                        # KeyError if no such key) -- NOTE that this
                        # access returns a *copy* of the entry!
        del d[key]      # delete data stored at key (raises KeyError
                        # if no such key)
        flag = key in d # true if the key exists
        list = d.keys() # a list of all existing keys (slow!)

        d.close()       # close it

Dependent on the implementation, closing a persistent dictionary may
or may not be necessary to flush changes to disk.

Normally, d[key] returns a COPY of the entry.  This needs care when
mutable entries are mutated: for example, if d[key] is a list,
        d[key].append(anitem)
does NOT modify the entry d[key] itself, as stored in the persistent
mapping -- it only modifies the copy, which is then immediately
discarded, so that the append has NO effect whatsoever.  To append an
item to d[key] in a way that will affect the persistent mapping, use:
        data = d[key]
        data.append(anitem)
        d[key] = data

To avoid the problem with mutable entries, you may pass the keyword
argument writeback=True in the call to shelve.open.  When you use:
        d = shelve.open(filename, writeback=True)
then d keeps a cache of all entries you access, and writes them all back
to the persistent mapping when you call d.close().  This ensures that
such usage as d[key].append(anitem) works as intended.

However, using keyword argument writeback=True may consume vast amount
of memory for the cache, and it may make d.close() very slow, if you
access many of d's entries after opening it in this way: d has no way to
check which of the entries you access are mutable and/or which ones you
actually mutate, so it must cache, and write back at close, all of the
entries that you access.  You can call d.sync() to write back all the
entries in the cache, and empty the cache (d.sync() also synchronizes
the persistent dictionary on disk, if feasible).
"""
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, loads, dumps
import zstandard as zstd
import dbm

import collections.abc

__all__ = ["Shelf", "BsdDbShelf", "DbfilenameShelf", "open"]


def read_pkl_zstd(dctx, obj):
    """
    Deserializer from a pickled object compressed with zstandard.

    Parameters
    ----------
    obj : bytes or str
        Either a bytes object that has been pickled and compressed or a str path to the file object.

    Returns
    -------
    Python object
    """
    obj1 = loads(dctx.decompress(obj))

    return obj1


def write_pkl_zstd(cctx, obj, compress_level=1, pkl_protocol=5):
    """
    Serializer using pickle and zstandard. Converts any object that can be pickled to a binary object then compresses it using zstandard. Optionally saves the object to disk. If obj is bytes, then it will only be compressed without pickling.

    Parameters
    ----------
    obj : any
        Any pickleable object.
    compress_level : int
        zstandard compression level.

    Returns
    -------
    If file_path is None, then it returns the byte object, else None.
    """
    c_obj = cctx.compress(dumps(obj, protocol=pkl_protocol))

    return c_obj


class _ClosedDict(collections.abc.MutableMapping):
    'Marker for a closed dict.  Access attempts raise a ValueError.'

    def closed(self, *args):
        raise ValueError('invalid operation on closed shelf')
    __iter__ = __len__ = __getitem__ = __setitem__ = __delitem__ = keys = closed

    def __repr__(self):
        return '<Closed Dictionary>'


class Shelf(collections.abc.MutableMapping):
    """Base class for shelf implementations.

    This is initialized with a dictionary-like object.
    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, dict, protocol=None, writeback=False,
                 keyencoding="utf-8", compressor='zstd', compress_level=1):
        self.dict = dict
        if protocol is None:
            protocol = 5
        self._protocol = protocol
        self.writeback = writeback
        self.cache = {}
        self.keyencoding = keyencoding
        self._compress_level = compress_level

        if compressor.lower() == 'zstd':
            self._compressor = zstd.ZstdCompressor(level=compress_level)
            self._decompressor = zstd.ZstdDecompressor()
        else:
            raise NotImplementedError('Only zstd is implemented.')

    def __iter__(self):
        for k in self.dict.keys():
            yield k.decode(self.keyencoding)

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        return key.encode(self.keyencoding) in self.dict

    def get(self, key, default=None):
        if key.encode(self.keyencoding) in self.dict:
            return self[key]
        return default

    def __getitem__(self, key):
        try:
            value = self.cache[key]
        except KeyError:
            # f = BytesIO(self.dict[key.encode(self.keyencoding)])
            value = read_pkl_zstd(self._decompressor, self.dict[key.encode(self.keyencoding)])
            if self.writeback:
                self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        if self.writeback:
            self.cache[key] = value
        p = write_pkl_zstd(self._compressor, value, pkl_protocol=self._protocol, compress_level=self._compress_level)
        self.dict[key.encode(self.keyencoding)] = p

    def __delitem__(self, key):
        del self.dict[key.encode(self.keyencoding)]
        try:
            del self.cache[key]
        except KeyError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dict is None:
            return
        try:
            self.sync()
            try:
                self.dict.close()
            except AttributeError:
                pass
        finally:
            # Catch errors that may happen when close is called from __del__
            # because CPython is in interpreter shutdown.
            try:
                self.dict = _ClosedDict()
            except:
                self.dict = None

    def __del__(self):
        if not hasattr(self, 'writeback'):
            # __init__ didn't succeed, so don't bother closing
            # see http://bugs.python.org/issue1339007 for details
            return
        self.close()

    def sync(self):
        if self.writeback and self.cache:
            self.writeback = False
            for key, entry in self.cache.items():
                self[key] = entry
            self.writeback = True
            self.cache = {}
        if hasattr(self.dict, 'sync'):
            self.dict.sync()


class BsdDbShelf(Shelf):
    """Shelf implementation using the "BSD" db interface.

    This adds methods first(), next(), previous(), last() and
    set_location() that have no counterpart in [g]dbm databases.

    The actual database must be opened using one of the "bsddb"
    modules "open" routines (i.e. bsddb.hashopen, bsddb.btopen or
    bsddb.rnopen) and passed to the constructor.

    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, dict, protocol=None, writeback=False,
                 keyencoding="utf-8"):
        Shelf.__init__(self, dict, protocol, writeback, keyencoding)

    def set_location(self, key):
        key, value = self.dict.set_location(key)
        return key.decode(self.keyencoding), read_pkl_zstd(self._decompressor, value)

    def next(self):
        key, value = next(self.dict)
        return key.decode(self.keyencoding), read_pkl_zstd(self._decompressor, value)

    def previous(self):
        key, value = self.dict.previous()
        return key.decode(self.keyencoding), read_pkl_zstd(self._decompressor, value)

    def first(self):
        key, value = self.dict.first()
        return key.decode(self.keyencoding), read_pkl_zstd(self._decompressor, value)

    def last(self):
        key, value = self.dict.last()
        return key.decode(self.keyencoding), read_pkl_zstd(self._decompressor, value)


class DbfilenameShelf(Shelf):
    """Shelf implementation using the "dbm" generic dbm interface.

    This is initialized with the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, filename, flag='c', protocol=None, writeback=False, compress_level=1):
        Shelf.__init__(self, dbm.open(str(filename), flag), protocol, writeback, compress_level=compress_level)


def open(filename, flag='c', protocol=5, writeback=False, compressor='zstd', compress_level=1):
    """Open a persistent dictionary for reading and writing. The values in the dictionary must be pickleable and will be compressed using the given compressor.

    Parameters
    -----------
    filename : str or pathlib.Path
        It must be a path to a local file location. Likely, multiple files will be created that represent the saved binary data (dat) and the index (dir).
    flag : str
        Flag associated with how the file is opened according to the dbm. See below for details.
    protocol : int
        The pickle protocol to use.
    writeback : bool
        If the writeback parameter is set to True, all entries accessed are also cached in memory, and written back on sync() and close().
    compressor : str
        The compressor to use to compress the pickle object before being written. Currently, only zstd is accepted.
    compress_level : int
        The compression level for the compressor.

    Returns
    -------
    DbfilenameShelf

    The optional *flag* argument can be:

   +---------+-------------------------------------------+
   | Value   | Meaning                                   |
   +=========+===========================================+
   | ``'r'`` | Open existing database for reading only   |
   |         | (default)                                 |
   +---------+-------------------------------------------+
   | ``'w'`` | Open existing database for reading and    |
   |         | writing                                   |
   +---------+-------------------------------------------+
   | ``'c'`` | Open database for reading and writing,    |
   |         | creating it if it doesn't exist           |
   +---------+-------------------------------------------+
   | ``'n'`` | Always create a new, empty database, open |
   |         | for reading and writing                   |
   +---------+-------------------------------------------+

    """

    return DbfilenameShelf(str(filename), flag, protocol, writeback, compress_level)


