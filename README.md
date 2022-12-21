# qg_probe
A simple Python module to exercise T5 and BART question generation models.

The heart of the module is the Probe class that caches the various models and tokenizers.

A question generation probe is instantiated with a single argument that is a string with the pathname
for the root of the model tree.
