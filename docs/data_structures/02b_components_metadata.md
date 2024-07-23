# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Metadata

A Thing can have *Metadata*; metadata are typically *labels* which add useful information.

Metadata for any given Thing takes the form of a key-value mapping of atomic elements or a single atomic element (FILEID) indicating a static set of metadata to use, explicitly tracked across versions of Things.

If a set of metadata is represented as a FILEID that FILEID is assumed to have a metadata component with compatible schema.

Metadata may not be nested and must contain atomic values.

Here are some abstract examples which you could use in Python.

Metadata are explicitly versioned across versions of Things.

```python
metadata = {'favorite_color': 'red', 'business_process': 'auto'}
```

Metadata is simply retrievable!

```python
metadata = thingstore.get_metadata('specific_file')
```
