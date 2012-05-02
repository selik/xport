========
Xport
========
------------------------------------------------------------
Python reader for SAS XPORT data transport files.
------------------------------------------------------------

What's it for?
==============

XPORT is the binary file format used by a bunch of `United States government agencies`_
for publishing data sets. It made a lot of sense if you were trying to read data files on your IBM mainframe back in 1988.

.. _United States government agencies: https://www.google.com/search?q=site:.gov+xpt+file

How do I use it?
================

Let's make this short and sweet::

    import xport
    with xport.XportReader(xport_file) as reader:
        for row in reader:
            print row

Each `row` will be a dict with a key for each field in the dataset. Values will be either a unicode string,
a float or an int, depending on the type specified in the file for that field.

Getting file info
=================

Once you have an `XportReader` object, there are a few properties and methods that will give you details about the file:

* reader.file: the underlying Python file object (see next section).

* reader.record_start: the position (in bytes) in the file where records start (see next section).

* reader.record_length: the length (in bytes) of each record (see next section).

* reader.record_count(): number of records in file. (Warning: this will seek to the end of the file to determine file length.)

* reader.file_info and reader.member_info: dicts containing information about when and how the dataset was created.

* reader.fields: list of fields in the dataset. Each field is a dict containing the following keys, copied from the spec::

    struct NAMESTR {
        short   ntype;              /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR    */
        short   nhfun;              /* HASH OF NNAME (always 0)            */
    *   short   field_length;       /* LENGTH OF VARIABLE IN OBSERVATION   */
        short   nvar0;              /* VARNUM                              */
    *   char8   name;              /* NAME OF VARIABLE                    */
    *   char40  label;             /* LABEL OF VARIABLE                   */

        char8   nform;              /* NAME OF FORMAT                      */
        short   nfl;                /* FORMAT FIELD LENGTH OR 0            */
    *   short   num_decimals;       /* FORMAT NUMBER OF DECIMALS           */
        short   nfj;                /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST  */
        char    nfill[2];           /* (UNUSED, FOR ALIGNMENT AND FUTURE)  */
        char8   niform;             /* NAME OF INPUT FORMAT                */
        short   nifl;               /* INFORMAT LENGTH ATTRIBUTE           */
        short   nifd;               /* INFORMAT NUMBER OF DECIMALS         */
        long    npos;               /* POSITION OF VALUE IN OBSERVATION    */
        char    rest[52];           /* remaining fields are irrelevant     */
        };

 **NOTE: items with stars have been renamed from the short names given in the spec.
 Since this is an alpha release, other items may be renamed in the future, if someone tells me what they're for.**

Random access to records
========================

If you want to access specific records, instead of iterating, you can use Python's standard file access
functions and a little math.

Get 1000th record::

    reader.file.seek(reader.record_start + reader.record_length * 1000, 0)
    reader.next()

Get record before most recent one fetched::

    reader.file.seek(-reader.record_length * 2, 1)
    reader.next()

Get last record::

    reader.file.seek(reader.record_start + reader.record_length * (reader.record_count() - 1), 0)
    reader.next()

(In this last example, note that we can't seek from the end of the file, because there may be padding bytes.
Good old fixed-width binary file formats.)

Please fix/steal this code!
===========================

I wrote this up because it seemed ridiculous that there was no easy way to read a standard government data format
in most programming languages. I may have gotten things wrong. If you find a file that doesn't decode propery,
send a pull request. `The official spec is here`_. It's surprisingly straightforward for a binary file format from the 80s.

.. _The official spec is here: http://support.sas.com/techsup/technote/ts140.html

Please also feel free to use this code as a base to write your own library for your favorite programming language.
Government data should be accessible, man.