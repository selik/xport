Change Log
==========

v0.1.0, 2012-05-02
  Initial release.

v0.2.0, 2016-03-22
  Major revision.

v0.2.0, 2016-03-23
  Add numpy and pandas converters.

v1.0.0, 2016-10-21
  Revise API to the pattern of from/to <format>

v2.0.0, 2016-10-21
  Reader yields regular tuples, not namedtuples

v3.0.0, 2020-04-20
  Revise API to the load/dump pattern.
  Enable specifying dataset name, variable names, labels, and formats.

v3.1.0, 2020-04-20
  Allow ``dumps(dataframe)`` instead of requiring a ``Dataset``.

v3.2.2, 2020-09-03
  Fix a bug that incorrectly displays a - (dash) when it's a null for numeric field.

v3.3.0, 2021-12-25
  Enable reading Transport Version 8/9 files.  Merry Christmas!

v3.4.0, 2021-12-25
  Add support for special missing values, like `.A`, that extend `float`.

v3.5.0, 2021-12-31
  Enable writing Transport Version 8 files.  Happy New Year!

v3.5.1, 2022-02-01
  Fix issues with writing ``Dataset.label`` and ``Variable.label``.

v3.6.0, 2022-02-02
  Add beta support for changing the text encoding for data and metadata.

v3.6.1, 2022-02-15
  Fix issue with v8 format when the dataset has no long labels.
